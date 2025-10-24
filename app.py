from flask import send_from_directory






import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import AzureOpenAI
from flask import Flask, request, jsonify
from datetime import datetime
import re

app = Flask(__name__)

class NLToSQL:
    def __init__(self, schema_json: dict, db_conn_string: str, azure_endpoint: str, azure_api_key: str, deployment_name: str):
        self.db_conn_string = db_conn_string
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version="2024-02-15-preview"
        )
        self.deployment = deployment_name
        self.schema = self._format_schema(schema_json)
    
    def _format_schema(self, schema_json: dict) -> str:
        """Format schema with enum values"""
        
        # Actual enum values from database
        enum_values = {
            'Issue': {
                'status': ['Fail', 'Pass', 'Suggestion', 'Validate'],
                'prod_status': ['Completed', 'Delete', 'In Progress', 'Not Started', 'To Do'],
                'updated_status': ['Fail', 'None', 'Pass'],
                'severity': ['Best Practice', 'Blocker', 'Critical', 'Major', 'Minor', 'NA'],
                'conformance_level': ['A', 'AA'],
                'type': ['error', 'notice', 'pass', 'warning'],
                'source': ['automated', 'extension']
            },
            'Page': {
                'scanStatus': ['completed', 'failed', 'failed_network', 'in_progress', 'not_started', 'pending', 'running']
            },
            'ScanResult': {
                'status': ['completed', 'failed', 'failed_network', 'in_progress', 'pending', 'running'],
                'batchStatus': ['completed', 'pending']
            }
        }
        
        schema_text = []
        for table_name, table_info in schema_json['tables'].items():
            cols = []
            for col in table_info['columns']:
                pk = " PRIMARY KEY" if col['is_primary_key'] else ""
                fk = " FOREIGN KEY" if col['is_foreign_key'] else ""
                nullable = " NULL" if col['is_nullable'] == 'YES' else " NOT NULL"
                
                # Add enum hint
                enum_hint = ""
                if table_name in enum_values and col['column_name'] in enum_values[table_name]:
                    values = enum_values[table_name][col['column_name']]
                    enum_hint = f" ENUM({', '.join(repr(v) for v in values)})"
                
                cols.append(f"  {col['column_name']} {col['data_type']}{pk}{fk}{nullable}{enum_hint}")
            
            if table_info.get('foreign_keys'):
                cols.append("\n  Foreign Keys:")
                for fk in table_info['foreign_keys']:
                    cols.append(f"    {fk['source_column']} -> {fk['target_table']}.{fk['target_column']}")
            
            schema_text.append(f"Table: {table_name}\n" + "\n".join(cols))
        
        return "\n\n".join(schema_text)
    
    def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language"""
        
        system_prompt = f"""You are a PostgreSQL expert. Generate ONLY SELECT queries.

    Database Schema:
    {self.schema}

    CRITICAL RULES:
    1. Use exact column/table names from schema - wrap ALL in double quotes
    2. PostgreSQL is case-sensitive: "Project", "Issue", "Page" (match schema exactly)
    3. For enum columns, use EXACT values shown in ENUM(...) - they are case-sensitive
    4. Project.id and Issue.project_id are TEXT type (not integer)
    5. Use table aliases: FROM "Issue" i JOIN "Page" p ON i."page_id" = p.id
    6. Always add LIMIT 100 unless user specifies otherwise
    7. Handle NULL properly: WHERE column IS NULL or WHERE column IS NOT NULL

    ENUM VALUE MAPPINGS (case-sensitive):
    - User says "fail/failed/failing" → WHERE "status" = 'Fail'
    - User says "pass/passed/passing" → WHERE "status" = 'Pass'
    - User says "suggestion/suggested" → WHERE "status" = 'Suggestion'
    - User says "validate/validation" → WHERE "status" = 'Validate'
    - User says "critical/high priority" → WHERE "severity" = 'Critical'
    - User says "major" → WHERE "severity" = 'Major'
    - User says "minor/low" → WHERE "severity" = 'Minor'

    COMMON QUERIES:
    - Count issues by project: SELECT p."name", COUNT(i.id) FROM "Project" p JOIN "Issue" i ON p.id = i."project_id" WHERE ... GROUP BY p."name"
    - Issues by status: WHERE i."status" = 'Fail' (not 'fail' or 'failed')
    - Recent issues: WHERE i."created_at" >= NOW() - INTERVAL '30 days'

    Return ONLY the SQL query - no markdown, no explanations."""

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        
        sql = response.choices[0].message.content.strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()
        sql = re.sub(r'\s+', ' ', sql)
        return sql
    
    def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """Comprehensive SQL validation"""
        sql_upper = sql.upper().strip()
        
        # Check for dangerous operations
        dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC']
        for keyword in dangerous:
            if keyword in sql_upper:
                return False, f"Forbidden operation: {keyword}"
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            return False, "Query must be a SELECT statement"
        
        # Check for SQL injection patterns
        suspicious = [';--', '/*', '*/', 'xp_', 'sp_', 'UNION ALL', 'UNION SELECT']
        for pattern in suspicious:
            if pattern.upper() in sql_upper:
                return False, f"Suspicious pattern detected: {pattern}"
        
        # Basic syntax validation
        if sql.count('(') != sql.count(')'):
            return False, "Unbalanced parentheses"
        
        return True, "Valid"
    
    def _execute(self, sql: str) -> tuple[list, dict]:
        """Execute SQL and return results with metadata"""
        conn = psycopg2.connect(self.db_conn_string)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Set statement timeout for safety (30 seconds)
                cursor.execute("SET statement_timeout = '30s'")
                
                # Execute query
                start_time = datetime.now()
                cursor.execute(sql)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                results = cursor.fetchall()
                
                metadata = {
                    "row_count": len(results),
                    "execution_time_seconds": round(execution_time, 3),
                    "columns": [desc[0] for desc in cursor.description] if cursor.description else []
                }
                
                return [dict(row) for row in results], metadata
        finally:
            conn.close()
    
    def _generate_human_response(self, question: str, sql: str, results: list, metadata: dict) -> dict:
        """Format results for stakeholders"""
        
        if not results:
            return {
                "summary": "No data found matching your query criteria.",
                "key_findings": [],
                "note": "Try adjusting your filters or check if data exists for this period."
            }
        
        # Show only first 20 rows to LLM
        preview = results[:20]
        
        system_prompt = """You are a business analyst presenting database query results.

    RULES:
    1. Use ONLY values from the results - never calculate anything yourself
    2. Present key findings clearly (3-5 points)
    3. Format numbers/dates as they appear
    4. Return JSON: {"summary": "...", "key_findings": ["...", "..."]}"""

        user_prompt = f"""Question: {question}

    Results ({metadata['row_count']} rows, showing first {len(preview)}):
    {json.dumps(preview, indent=2, default=str)}

    Present these results professionally."""

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def query(self, question: str) -> dict:
        """Main method: NL question -> SQL + Results + Human Response"""
        sql = None
        try:
            # Step 1: Generate SQL
            sql = self._generate_sql(question)
            
            # Step 2: Validate SQL
            is_valid, validation_msg = self._validate_sql(sql)
            if not is_valid:
                raise ValueError(f"SQL validation failed: {validation_msg}")
            
            # Step 3: Execute SQL
            results, metadata = self._execute(sql)
            
            # Step 4: Generate human-readable response
            human_response = self._generate_human_response(question, sql, results, metadata)
            
            return {
                "success": True,
                "question": question,
                "sql": sql,
                "results": results,
                "metadata": metadata,
                "response": human_response,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except psycopg2.Error as e:
            return {
                "success": False,
                "error": f"Database error: {str(e)}",
                "error_type": "database",
                "sql": sql,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "general",
                "sql": sql,
                "timestamp": datetime.utcnow().isoformat()
            }


# Global instance
nl_sql_instance = None

def init_nl_sql(schema_json: dict):
    """Initialize NLToSQL with schema"""
    global nl_sql_instance
    
    DB_CONN = "postgresql://postgres:nidshwngisgnxtab@74.208.202.238:5444/a11y-now-master"
    AZURE_ENDPOINT = "https://ai-barrierbreak6904ai389817507120.openai.azure.com/"
    AZURE_API_KEY = "3ZwtHD7mcTuGgnfvVemTX49aqtQxaq62Jcshb2Yrgmux4xcXEbFWJQQJ99ALACHYHv6XJ3w3AAAAACOG08Rb"
    DEPLOYMENT = "gpt-4o"

    # Validate environment variables
    missing = []
    if not DB_CONN: missing.append("DB_CONNECTION_STRING")
    if not AZURE_ENDPOINT: missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_API_KEY: missing.append("AZURE_OPENAI_KEY")
    if not DEPLOYMENT: missing.append("AZURE_DEPLOYMENT_NAME")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    nl_sql_instance = NLToSQL(schema_json, DB_CONN, AZURE_ENDPOINT, AZURE_API_KEY, DEPLOYMENT)
    print(f"✓ NL to SQL service initialized with deployment: {DEPLOYMENT}")


@app.route('/')
def serve_index():
    """Serve frontend"""
    return send_from_directory('.', 'index.html')

# API Routes
@app.route('/api/query', methods=['POST'])
def query_endpoint():
    """Main query endpoint"""
    if not nl_sql_instance:
        return jsonify({
            "success": False, 
            "error": "Service not initialized. Check environment variables."
        }), 500
    
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({
            "success": False, 
            "error": "Missing 'question' in request body"
        }), 400
    
    question = data['question'].strip()
    if not question:
        return jsonify({
            "success": False, 
            "error": "Question cannot be empty"
        }), 400
    
    result = nl_sql_instance.query(question)
    
    status_code = 200 if result['success'] else 400
    return jsonify(result), status_code


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    db_status = "unknown"
    
    if nl_sql_instance:
        try:
            conn = psycopg2.connect(nl_sql_instance.db_conn_string)
            conn.close()
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy" if nl_sql_instance else "not_initialized",
        "service_initialized": nl_sql_instance is not None,
        "database_status": db_status,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/api/schema', methods=['GET'])
def get_schema():
    """Return available tables and columns"""
    if not nl_sql_instance:
        return jsonify({"success": False, "error": "Service not initialized"}), 500
    
    return jsonify({
        "success": True,
        "schema": nl_sql_instance.schema
    })


if __name__ == "__main__":
    try:
        # Load schema
        with open('metadata.json', 'r') as f:
            schema_data = json.load(f)
        
        print("✓ Schema loaded successfully")
        
        # Initialize service
        init_nl_sql(schema_data)
        
        # Run Flask app
        port = int(os.getenv("PORT", 5005))
        print(f"\n🚀 Starting server on port {port}...")
        print(f"   Health check: http://localhost:{port}/api/health")
        print(f"   Query endpoint: http://localhost:{port}/api/query")
        
        app.run(host='0.0.0.0', port=port, debug=True)
        
    except Exception as e:
        print(f"❌ Startup failed: {str(e)}")
        exit(1)
