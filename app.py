from flask import Flask, jsonify, request, send_from_directory, session
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import requests
import json
import time
import re
import os
from flask_cors import CORS
from flask_session import Session

app = Flask(__name__)

cors_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://ai-test-aitest-lef0n4-8c8462-74-208-202-238.traefik.me",
    "https://ai-test-aitest-lef0n4-8c8462-74-208-202-238.traefik.me"  # add this
]


CORS(app, 
     supports_credentials=True, 
     origins=cors_origins,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "BarrierBreak")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True

Session(app)

EXAMPLE_ROWS = 3
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 6

METADATA_CACHE = {}
SCHEMA_REGISTRY = {
    "tables": {},
    "foreign_keys": [],
    "loaded": False
}


# ============================================================================
# DATABASE & CORE FUNCTIONS
# ============================================================================

def get_db_connection(conn_str):
    """Create database connection with explicit connection string"""
    try:
        return psycopg2.connect(conn_str)
    except psycopg2.Error as e:
        raise Exception(f"Database connection failed: {e}")


def load_schema_registry(conn_str, schema_name="public"):
    """Load schema with explicit connection string"""
    if SCHEMA_REGISTRY["loaded"]:
        return SCHEMA_REGISTRY
    
    print("Loading schema registry...")
    
    conn = get_db_connection(conn_str)
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_type = 'BASE TABLE'
            AND table_name != '_prisma_migrations'
            ORDER BY table_name
        """, (schema_name,))
        
        tables = [row[0] for row in cur.fetchall()]
        
        cur.execute("""
            SELECT
                tc.table_name as source_table,
                kcu.column_name as source_column,
                ccu.table_name as target_table,
                ccu.column_name as target_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s
        """, (schema_name,))
        
        foreign_keys = [
            {
                "source_table": row[0],
                "source_column": row[1],
                "target_table": row[2],
                "target_column": row[3]
            }
            for row in cur.fetchall()
        ]
        
        SCHEMA_REGISTRY["tables"] = {table: None for table in tables}
        SCHEMA_REGISTRY["foreign_keys"] = foreign_keys
        SCHEMA_REGISTRY["loaded"] = True
        
        print(f"Schema registry loaded: {len(tables)} tables, {len(foreign_keys)} foreign keys")
        return SCHEMA_REGISTRY
        
    finally:
        cur.close()
        conn.close()


def find_table_relationships(table_names):
    """Find relationships between tables (schema already loaded)"""
    if not SCHEMA_REGISTRY["loaded"]:
        raise Exception("Schema registry not loaded")
    
    table_set = set(table_names)
    relevant_fks = []
    
    for fk in SCHEMA_REGISTRY["foreign_keys"]:
        if fk["source_table"] in table_set and fk["target_table"] in table_set:
            relevant_fks.append(fk)
    
    return relevant_fks


def detect_tables_from_query(user_query):
    """Detect tables from natural language query"""
    if not SCHEMA_REGISTRY["loaded"]:
        raise Exception("Schema registry not loaded")
    
    query_lower = user_query.lower()
    detected_tables = set()
    
    all_tables = list(SCHEMA_REGISTRY["tables"].keys())
    
    table_keywords = {
        "Issue": ["issue", "issues", "violation", "violations", "failure", "failures", "bug", "bugs", "problem", "problems"],
        "Project": ["project", "projects"],
        "Page": ["page", "pages", "url", "urls"],
        "ScanResult": ["scan result", "scan results", "scanning", "scan"],
        "AccessibilityStandard": ["accessibility standard", "wcag", "guideline"],
        "Activity": ["activity", "activities"],
        "Comment": ["comment", "comments"],
        "ProjectToUser": ["assigned user", "user assignment", "assigned to", "project assignment"],
        "UserToRole": ["user role", "role assignment"],
        "Role": ["role", "roles"],
    }
    
    for table, keywords in table_keywords.items():
        if table in all_tables:
            for keyword in keywords:
                if f" {keyword} " in f" {query_lower} " or query_lower.startswith(keyword) or query_lower.endswith(keyword):
                    detected_tables.add(table)
                    break
    
    if any(word in query_lower for word in ["user", "users"]):
        if any(word in query_lower for word in ["project", "projects"]):
            detected_tables.add("ProjectToUser")
            detected_tables.add("Project")
        if any(word in query_lower for word in ["role", "roles"]):
            detected_tables.add("UserToRole")
            detected_tables.add("Role")
    
    if not detected_tables:
        detected_tables.add("Issue")
    
    detected_list = list(detected_tables)
    if len(detected_list) > 4:
        priority = ["Issue", "Project", "Page", "ProjectToUser"]
        detected_list = [t for t in priority if t in detected_list][:4]
    
    return detected_list


def call_azure_llm_with_retry(prompt, azure_key, azure_endpoint, max_tokens=2000, temperature=0.0):
    """Call Azure LLM with explicit credentials"""
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_key
    }
    
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                azure_endpoint, 
                headers=headers, 
                json=data, 
                timeout=45
            )
            
            if response.status_code == 200:
                return response.json()
            
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    retry_delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    print(f"Rate limited. Waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {MAX_RETRIES} attempts")
            
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                print(f"Timeout. Retrying in {INITIAL_RETRY_DELAY}s...")
                time.sleep(INITIAL_RETRY_DELAY)
                continue
            raise Exception("Request timed out after multiple attempts")
        
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Server error. Retrying...")
                time.sleep(INITIAL_RETRY_DELAY)
                continue
            raise Exception(f"LLM API call failed: {str(e)}")
    
    raise Exception("Failed to get response from LLM after all retries")


def get_table_structure_and_count(conn_str, schema_name, table_name, include_examples=True):
    """Get table structure with explicit connection"""
    conn = get_db_connection(conn_str)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """, (schema_name, table_name))
        
        columns_info = [
            {"name": row[0], "type": row[1], "nullable": row[2]} 
            for row in cur.fetchall()
        ]

        if not columns_info:
            raise Exception(f"Table {schema_name}.{table_name} not found")

        count_query = sql.SQL('SELECT COUNT(*) FROM {}.{}').format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name)
        )
        cur.execute(count_query)
        row_count = cur.fetchone()[0]

        if include_examples:
            for col in columns_info:
                example_query = sql.SQL(
                    'SELECT {} FROM {}.{} WHERE {} IS NOT NULL LIMIT %s'
                ).format(
                    sql.Identifier(col["name"]),
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.Identifier(col["name"])
                )
                cur.execute(example_query, (EXAMPLE_ROWS,))
                col["examples"] = [row[0] for row in cur.fetchall()]

        return {"columns": columns_info, "row_count": row_count}

    finally:
        cur.close()
        conn.close()


def call_llm_with_metadata(table_name, metadata, azure_key, azure_endpoint):
    """Generate metadata insights"""
    prompt = f"""You are analyzing a PostgreSQL table. Generate a JSON response with:
1. "summary": Brief description of what this table stores
2. "column_synonyms": Object mapping each column name to array of 3 alternative names users might use
3. "sample_queries": Array of 5-8 natural language questions users could ask
4. "sql_hints": Useful notes about common groupings, filters, or aggregations

Table: {table_name}
Row Count: {metadata['row_count']}

Columns:
"""
    for col in metadata['columns']:
        examples = col.get('examples', [])
        prompt += f"- {col['name']} ({col['type']}): {examples[:3]}\n"

    prompt += "\nRespond ONLY with valid JSON, no markdown formatting."

    try:
        result = call_azure_llm_with_retry(prompt, azure_key, azure_endpoint, max_tokens=2000, temperature=0.1)
        content = result['choices'][0]['message']['content'].strip()
        
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        return json.loads(content)

    except json.JSONDecodeError as e:
        return {"error": "Failed to parse LLM response", "details": str(e)}
    except Exception as e:
        return {"error": "LLM request failed", "details": str(e)}


def generate_multi_table_sql(schema_name, table_names, user_query, all_metadata, all_insights, azure_key, azure_endpoint):
    """Generate SQL from natural language"""
    relationships = find_table_relationships(table_names)
    
    def needs_quoting(name):
        if not name:
            return False
        return (name != name.lower() or not name[0].isalpha() or 
                not all(c.isalnum() or c == '_' for c in name))
    
    tables_info = []
    for table_name in table_names:
        metadata = all_metadata.get(table_name, {})
        insights = all_insights.get(table_name, {})
        
        if not metadata:
            continue
        
        table_ref = f'"{table_name}"' if needs_quoting(table_name) else table_name
        
        columns_info = []
        for col in metadata.get('columns', []):
            col_name = f'"{col["name"]}"' if needs_quoting(col["name"]) else col["name"]
            col_type = col['type']
            
            examples = col.get('examples', [])
            if examples:
                if col_type in ['integer', 'bigint', 'numeric', 'double precision']:
                    examples_str = ', '.join([str(ex) for ex in examples[:3]])
                else:
                    examples_str = ', '.join([f"'{str(ex)[:30]}'" for ex in examples[:3]])
            else:
                examples_str = "No examples"
            
            col_info = f"    • {col_name} ({col_type}): {examples_str}"
            
            if insights and 'column_synonyms' in insights:
                synonyms = insights['column_synonyms'].get(col['name'], [])
                if synonyms:
                    col_info += f" [Also: {', '.join(synonyms[:2])}]"
            
            columns_info.append(col_info)
        
        table_section = f"""  {table_ref}:
{chr(10).join(columns_info)}"""
        
        tables_info.append(table_section)
    
    relationships_info = []
    for rel in relationships:
        src_table = f'"{rel["source_table"]}"' if needs_quoting(rel["source_table"]) else rel["source_table"]
        tgt_table = f'"{rel["target_table"]}"' if needs_quoting(rel["target_table"]) else rel["target_table"]
        src_col = f'"{rel["source_column"]}"' if needs_quoting(rel["source_column"]) else rel["source_column"]
        tgt_col = f'"{rel["target_column"]}"' if needs_quoting(rel["target_column"]) else rel["target_column"]
        
        relationships_info.append(f"  • {src_table}.{src_col} → {tgt_table}.{tgt_col}")
    
    prompt = f"""You are a PostgreSQL expert. Convert natural language to SQL query with JOINs if needed.

DATABASE CONTEXT:
Schema: {schema_name}
Available Tables: {len(table_names)}

TABLES (use EXACT syntax shown):
{chr(10).join(tables_info)}

TABLE RELATIONSHIPS:
{chr(10).join(relationships_info) if relationships_info else "  • No relationships needed (single table query)"}

USER REQUEST:
"{user_query}"

CRITICAL RULES:
1. Use ONLY column names explicitly listed in schemas above
2. Use ONLY table names from Available Tables
3. Generate ONLY the SELECT statement, no semicolon
4. Use table aliases and proper JOIN syntax
5. For "top N" queries: GROUP BY + COUNT + ORDER BY DESC + LIMIT N

Return ONLY the SQL query, no explanations."""

    try:
        result = call_azure_llm_with_retry(prompt, azure_key, azure_endpoint, max_tokens=2000, temperature=0.0)
        sql_query = result['choices'][0]['message']['content'].strip()
        
        if sql_query.startswith("```"):
            lines = sql_query.split('\n')
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            sql_query = '\n'.join(lines).strip()
        
        sql_query = sql_query.rstrip(';').strip()
        
        if not sql_query.upper().startswith('SELECT'):
            raise Exception(f"LLM did not generate SQL. Response: {sql_query[:200]}")
        
        return sql_query
    
    except Exception as e:
        raise Exception(f"Multi-table SQL generation failed: {str(e)}")


def analyze_query_results(sql_query, results):
    """Analyze query results structure"""
    if not results:
        return {
            "query_type": "empty",
            "total_results": 0,
            "summary": {"message": "No results found"}
        }
    
    sql_upper = sql_query.upper()
    total_results = len(results)
    first_row = results[0]
    
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', sql_upper))
    
    agg_candidates = {}
    for key, value in first_row.items():
        if isinstance(value, (int, float)) and not key.endswith('_id'):
            agg_candidates[key] = True
    
    if total_results == 1 and len(first_row) == 1:
        col_name = list(first_row.keys())[0]
        col_value = first_row[col_name]
        if isinstance(col_value, (int, float)):
            return {
                "query_type": "simple_count",
                "total_results": 1,
                "summary": {
                    "total_count": col_value,
                    "count_column": col_name
                }
            }
    
    if has_group_by or (len(agg_candidates) > 0 and total_results > 1):
        agg_col = None
        
        for key in agg_candidates.keys():
            key_lower = key.lower()
            if any(x in key_lower for x in ['count', 'total', 'sum', 'avg', 'average', 'num']):
                agg_col = key
                break
        
        if not agg_col and agg_candidates:
            agg_col = list(agg_candidates.keys())[0]
        
        if agg_col:
            total_agg = sum(row[agg_col] for row in results)
            
            enriched = []
            for row in results:
                enriched_row = dict(row)
                agg_value = row[agg_col]
                enriched_row['percentage'] = round((agg_value / total_agg * 100), 2) if total_agg > 0 else 0
                enriched.append(enriched_row)
            
            enriched.sort(key=lambda x: x[agg_col], reverse=True)
            
            grouping_cols = [k for k in first_row.keys() if k != agg_col]
            
            return {
                "query_type": "aggregation_grouped",
                "total_results": total_results,
                "summary": {
                    "total_aggregated_value": total_agg,
                    "aggregation_column": agg_col,
                    "grouping_columns": grouping_cols,
                    "enriched_results": enriched
                }
            }
    
    return {
        "query_type": "simple_select",
        "total_results": total_results,
        "summary": {
            "columns": list(first_row.keys())
        }
    }


def determine_narrative_scope(total_results):
    """Determine how many results to include in narrative"""
    if total_results <= 100:
        return (total_results, total_results, "all_items")
    else:
        return (100, 50, "top_100_plus_summary")


def generate_narrative(user_query, analysis, sql_query, azure_key, azure_endpoint):
    """Generate narrative explanation of results"""
    query_type = analysis["query_type"]
    total_results = analysis["total_results"]
    
    if query_type == "empty":
        return {"formatted": "**No Results Found**\n\nNo data matched your query criteria.", "format": "markdown"}
    
    if query_type == "simple_count":
        count_val = analysis["summary"]["total_count"]
        count_col = analysis["summary"]["count_column"]
        
        prompt = f"""Generate a clear answer to: "{user_query}"

Result: The {count_col} is {count_val:,}

Write a natural 2-3 sentence response using this exact number."""
        
        try:
            result = call_azure_llm_with_retry(prompt, azure_key, azure_endpoint, max_tokens=2000, temperature=0.1)
            narrative = result['choices'][0]['message']['content'].strip()
        except:
            narrative = f"The {count_col} is **{count_val:,}**."
        
        return {"formatted": narrative, "format": "markdown"}
    
    if query_type == "aggregation_grouped":
        narrative_limit, preview_size, scope = determine_narrative_scope(total_results)
        
        summary = analysis["summary"]
        enriched_results = summary["enriched_results"]
        agg_col = summary["aggregation_column"]
        grouping_cols = summary["grouping_columns"]
        total_agg = summary["total_aggregated_value"]
        
        top_items = enriched_results[:narrative_limit]
        
        items_text = []
        for i, item in enumerate(top_items, 1):
            group_parts = []
            for col in grouping_cols:
                val = item[col]
                if val is None:
                    val = "NULL"
                elif isinstance(val, str) and len(val) > 50:
                    val = val[:50] + "..."
                group_parts.append(f"{val}")
            
            group_str = ", ".join(group_parts) if len(group_parts) > 1 else group_parts[0] if group_parts else "Unknown"
            count_value = item[agg_col]
            percentage = item['percentage']
            items_text.append(f"{i}. {group_str}: {count_value:,} ({percentage}%)")
        
        prompt = f"""Generate a formatted answer for: "{user_query}"

Total {agg_col.replace('_', ' ')}: {total_agg:,}
Categories: {total_results}

Top {narrative_limit} results:
{chr(10).join(items_text)}

Format:
1. Summary paragraph with **bold** key numbers
2. ### Top Results heading
3. Numbered list of results
4. Use markdown: **bold**, headings (###)

Return ONLY formatted markdown text."""
        
        try:
            result = call_azure_llm_with_retry(prompt, azure_key, azure_endpoint, max_tokens=2000, temperature=0.1)
            narrative = result['choices'][0]['message']['content'].strip()
        except:
            narrative = f"Found **{total_agg:,}** across **{total_results}** categories."
        
        return {"formatted": narrative, "format": "markdown"}
    
    if query_type == "simple_select":
        columns = analysis["summary"]["columns"]
        
        prompt = f"""Generate answer for: "{user_query}"

Found {total_results} records with columns: {', '.join(columns)}

Write 2-3 sentences. Use **bold** for count. Return markdown only."""
        
        try:
            result = call_azure_llm_with_retry(prompt, azure_key, azure_endpoint, max_tokens=2000, temperature=0.1)
            narrative = result['choices'][0]['message']['content'].strip()
        except:
            narrative = f"Found **{total_results}** matching records."
        
        return {"formatted": narrative, "format": "markdown"}
    
    return {"formatted": "**No Results Found**", "format": "markdown"}


def is_safe_sql(sql_query):
    """Verify SQL is safe (SELECT only)"""
    sql_upper = sql_query.upper().strip()
    
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT queries are allowed"
    
    dangerous_patterns = [
        r'\bDELETE\b', r'\bDROP\b', r'\bTRUNCATE\b', r'\bINSERT\b',
        r'\bUPDATE\s+', r'\bALTER\b', r'\bCREATE\s+', r'\bREPLACE\b',
        r'\bGRANT\b', r'\bREVOKE\b', r'\bEXECUTE\b', r'\bCALL\b',
        r'\bCOPY\b', r'\bIMPORT\b'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_upper):
            if pattern == r'\bCREATE\s+' and 'CREATED' in sql_upper:
                continue
            elif pattern == r'\bUPDATE\s+' and 'UPDATED' in sql_upper:
                continue
            else:
                return False, "Query contains forbidden keyword"
    
    return True, "Query is safe"


def execute_generated_sql(conn_str, generated_sql, max_rows=1000):
    """Execute SQL safely"""
    is_safe, message = is_safe_sql(generated_sql)
    if not is_safe:
        raise Exception(f"Unsafe SQL: {message}")
    
    conn = get_db_connection(conn_str)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        sql_upper = generated_sql.upper()
        if 'LIMIT' not in sql_upper:
            generated_sql += f" LIMIT {max_rows}"
        
        start_time = time.time()
        cur.execute(generated_sql)
        results = cur.fetchall()
        execution_time = time.time() - start_time
        
        return {
            "rows": [dict(row) for row in results],
            "execution_time_ms": round(execution_time * 1000, 2)
        }
    
    except psycopg2.Error as e:
        raise Exception(f"SQL execution failed: {e.pgerror or str(e)}")
    
    finally:
        cur.close()
        conn.close()


def paginate_results(results, page=1, page_size=50):
    """Paginate result set"""
    total = len(results)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        "total_results": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "results": results[start:end],
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "previous_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < total_pages else None
    }


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/login", methods=["POST"])
def login_route():
    """Store credentials in session"""
    data = request.get_json() or {}
    conn_str = data.get("conn_str")
    azure_endpoint = data.get("azure_endpoint")
    azure_key = data.get("azure_key")

    if not all([conn_str, azure_endpoint, azure_key]):
        return jsonify({"error": "conn_str, azure_endpoint and azure_key are required"}), 400

    try:
        test_conn = psycopg2.connect(conn_str, connect_timeout=5)
        test_conn.close()
    except Exception as e:
        return jsonify({"error": "Postgres connection failed", "details": str(e)}), 400

    try:
        headers = {"Content-Type": "application/json", "api-key": azure_key}
        payload = {
            "messages": [{"role": "system", "content": "health check"}],
            "max_tokens": 1,
            "temperature": 0.0
        }
        resp = requests.post(azure_endpoint, headers=headers, json=payload, timeout=8)
        
        if resp.status_code >= 500:
            return jsonify({"error": "Azure endpoint error"}), 400
        if resp.status_code == 401:
            return jsonify({"error": "Azure authentication failed"}), 400
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Azure endpoint check failed", "details": str(e)}), 400

    session["conn_str"] = conn_str
    session["azure_endpoint"] = azure_endpoint
    session["azure_key"] = azure_key

    return jsonify({"message": "Credentials stored in session"}), 200


@app.route("/query", methods=["POST"])
def query_route():
    """Standard query endpoint"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    
    user_query = data.get("query")
    schema = data.get("schema", "public")
    use_insights = data.get("use_insights", True)
    execute = data.get("execute", True)
    page = data.get("page", 1)
    force_refresh = data.get("force_refresh", False)
    explicit_tables = data.get("tables")
    
    if not user_query:
        return jsonify({"error": "query is required"}), 400

    # Hardcoded credentials
    conn_str = os.getenv("DATABASE_URL", "postgresql://postgres:nidshwngisgnxtab@74.208.202.238:5444/a11y-now-master")
    azure_key = os.getenv("AZURE_KEY", "3ZwtHD7mcTuGgnfvVemTX49aqtQxaq62Jcshb2Yrgmux4xcXEbFWJQQJ99ALACHYHv6XJ3w3AAAAACOG08Rb")
    azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://ai-barrierbreak6904ai389817507120.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview")
    
    if not all([conn_str, azure_key, azure_endpoint]):
        return jsonify({"error": "Server configuration error"}), 500

    try:
        if not SCHEMA_REGISTRY["loaded"]:
            load_schema_registry(conn_str, schema)
        
        if explicit_tables:
            table_names = explicit_tables
        else:
            table_names = detect_tables_from_query(user_query)
        
        all_metadata = {}
        all_insights = {}
        
        for table_name in table_names:
            cache_key = f"{schema}.{table_name}"
            
            if cache_key in METADATA_CACHE and not force_refresh:
                cached = METADATA_CACHE[cache_key]
                all_metadata[table_name] = cached["metadata"]
                all_insights[table_name] = cached.get("insights") if use_insights else None
            else:
                metadata = get_table_structure_and_count(conn_str, schema, table_name, include_examples=True)
                insights = None
                
                if use_insights:
                    insights = call_llm_with_metadata(table_name, metadata, azure_key, azure_endpoint)
                
                METADATA_CACHE[cache_key] = {
                    "metadata": metadata,
                    "insights": insights
                }
                
                all_metadata[table_name] = metadata
                all_insights[table_name] = insights
        
        generated_sql = generate_multi_table_sql(
            schema, table_names, user_query, 
            all_metadata, all_insights, azure_key, azure_endpoint
        )
        
        response = {
            "user_query": user_query,
            "generated_sql": generated_sql,
            "tables_used": table_names,
            "cache_used": any(f"{schema}.{t}" in METADATA_CACHE for t in table_names) and not force_refresh
        }
        
        if execute:
            try:
                result = execute_generated_sql(conn_str, generated_sql)
                raw_results = result["rows"]
                execution_time = result["execution_time_ms"]
                
                analysis = analyze_query_results(generated_sql, raw_results)
                
                if analysis["query_type"] == "aggregation_grouped":
                    processed_results = analysis["summary"]["enriched_results"]
                else:
                    processed_results = raw_results
                
                narrative = generate_narrative(user_query, analysis, generated_sql, azure_key, azure_endpoint)
                
                _, preview_size, scope = determine_narrative_scope(analysis["total_results"])
                
                pagination = paginate_results(processed_results, page, preview_size)
                
                response.update({
                    "narrative": narrative,
                    "data": pagination,
                    "metadata": {
                        "execution_time_ms": execution_time,
                        "query_type": analysis["query_type"],
                        "joins_used": len(table_names) > 1
                    },
                    "executed": True
                })
                
            except Exception as exec_error:
                response["executed"] = False
                response["execution_error"] = str(exec_error)
                response["note"] = "SQL generated but execution failed"
        else:
            response["executed"] = False
            response["note"] = "Set 'execute': true to run the query"
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "user_query": user_query,
            "tables_detected": table_names if 'table_names' in locals() else None,
            "generated_sql": generated_sql if 'generated_sql' in locals() else None
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
