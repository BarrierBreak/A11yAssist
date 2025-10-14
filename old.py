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

# CORS configuration with credentials support
CORS(app, supports_credentials=True, origins=["*"])

# Session configuration
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "BarrierBreak-change-in-production")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "/tmp/flask_session"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # Changed from None for local dev
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS

Session(app)

# Configuration
EXAMPLE_ROWS = 3

# Retry configuration for rate limiting
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 6

# In-memory cache for table metadata and insights
METADATA_CACHE = {}

# Schema registry - stores all table relationships
SCHEMA_REGISTRY = {
    "tables": {},
    "foreign_keys": [],
    "loaded": False
}

def get_db_connection():
    """Create database connection using runtime credential from session."""
    conn_str = session.get("conn_str")
    if not conn_str:
        raise Exception("Missing DB connection string. Please POST /login with credentials first.")
    try:
        return psycopg2.connect(conn_str)
    except psycopg2.Error as e:
        raise Exception(f"Database connection failed: {e}")


def load_schema_registry(schema_name="public"):
    """Load complete schema metadata for all tables."""
    if SCHEMA_REGISTRY["loaded"]:
        return SCHEMA_REGISTRY
    
    print("Loading schema registry...")
    
    conn = get_db_connection()
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
    """Find foreign key relationships between specified tables."""
    if not SCHEMA_REGISTRY["loaded"]:
        load_schema_registry()
    
    table_set = set(table_names)
    relevant_fks = []
    
    for fk in SCHEMA_REGISTRY["foreign_keys"]:
        if fk["source_table"] in table_set and fk["target_table"] in table_set:
            relevant_fks.append(fk)
    
    return relevant_fks


def detect_tables_from_query(user_query):
    """Detect which tables are needed based on user query."""
    if not SCHEMA_REGISTRY["loaded"]:
        load_schema_registry()
    
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


def call_azure_llm_with_retry(prompt, max_tokens=2000, temperature=0.0):
    """Call Azure OpenAI with retry mechanism for rate limiting."""
    azure_key = session.get("azure_key")
    azure_endpoint = session.get("azure_endpoint")
    if not azure_key or not azure_endpoint:
        raise Exception("Missing Azure credentials. Please POST /login with credentials first.")

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
                    print(f"⚠️  Rate limited. Waiting {retry_delay}s before retry... (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {MAX_RETRIES} attempts. Please wait a few minutes before trying again.")
            
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait_time = INITIAL_RETRY_DELAY
                print(f"⚠️  Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue
            raise Exception("Request timed out after multiple attempts")
        
        except requests.exceptions.RequestException as e:
            if response.status_code not in [429, 500, 502, 503, 504]:
                raise Exception(f"LLM API call failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                wait_time = INITIAL_RETRY_DELAY
                print(f"⚠️  Server error. Retrying in {wait_time}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue
            raise Exception(f"LLM API call failed after retries: {str(e)}")
    
    raise Exception("Failed to get response from LLM after all retries")


def get_table_structure_and_count(schema_name, table_name, include_examples=True):
    """Get table metadata with proper SQL injection protection"""
    conn = get_db_connection()
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


def call_llm_with_metadata(table_name, metadata):
    """Generate table insights using LLM"""
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
        result = call_azure_llm_with_retry(prompt, max_tokens=2000, temperature=0.1)
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


def generate_multi_table_sql(schema_name, table_names, user_query, all_metadata, all_insights):
    """Generate SQL for multi-table queries with JOINs."""
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Schema: {schema_name}
Available Tables: {len(table_names)}

TABLES (use EXACT syntax shown):
{chr(10).join(tables_info)}

TABLE RELATIONSHIPS:
{chr(10).join(relationships_info) if relationships_info else "  • No relationships needed (single table query)"}

USER REQUEST:
"{user_query}"

CRITICAL RULES FOR MULTI-TABLE SQL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0. MOST IMPORTANT - COLUMN NAMES:
   - You MUST ONLY use column names that are EXPLICITLY listed in the table schemas above
   - DO NOT invent, guess, or assume column names exist
   - If you need a column that doesn't exist, the query cannot be answered
   - Example: If "recommendation_text" is not in the schema, DO NOT use it
   - Double-check EVERY column name against the schemas before using it

1. TABLE NAMES:
   - ONLY use table names shown in "Available Tables" section
   - DO NOT invent table names (e.g., no "AccessibilityIssue" - use "Issue")
   - Use EXACT table name spelling and quoting as shown

2. TABLE SELECTION:
   - Only use tables that are necessary to answer the query
   - If query can be answered with 1 table, don't add unnecessary JOINs
   - Maximum 3 table JOINs for this query

3. JOIN SYNTAX:
   - Use INNER JOIN by default
   - Use LEFT JOIN only if explicitly needed (e.g., "including projects with no issues")
   - Always specify ON condition using the relationships shown above
   - Use table aliases for clarity (e.g., i for Issue, p for Project)
   
   Example:
   FROM "Issue" i
   INNER JOIN "Project" p ON i.project_id = p.id

4. COLUMN REFERENCES:
   - Prefix all columns with table alias (e.g., i.severity, p.name)
   - Use ONLY column names that exist in the schemas
   - Preserve quote style shown in schemas

5. JUNCTION TABLES (Many-to-Many Relationships):
   - ProjectToUser links Project ↔ User (with roleId)
   - UserToRole links User ↔ Role
   - Use these for queries about "users assigned to projects" or "user roles"
   
   Example: "projects with assigned users and roles"
   SELECT p.name, ptu.userId, ptu.roleId
   FROM "Project" p
   INNER JOIN "ProjectToUser" ptu ON p.id = ptu.projectId

6. "TOP N" QUERIES (CRITICAL):
   - "top N items" → GROUP BY + COUNT + ORDER BY DESC + LIMIT N
   - "top N in category" → Add WHERE for category + GROUP BY + COUNT + ORDER BY + LIMIT
   
   Example:
   "top 5 projects with most issues" →
   SELECT p.id, COUNT(*) AS issue_count
   FROM "Issue" i
   INNER JOIN "Project" p ON i.project_id = p.id
   GROUP BY p.id
   ORDER BY issue_count DESC
   LIMIT 5

7. HANDLING MISSING COLUMNS:
   - If the query asks for data that doesn't exist in any column, return a simple query showing available data
   - Example: Query asks for "recommendation text" but no such column exists
   - Generate: SELECT id, severity, status FROM "Issue" LIMIT 10
   - The narrative layer will handle explaining what's available

8. AGGREGATIONS ACROSS TABLES:
   - Group by columns from the dimension table
   - COUNT(*) counts rows from the fact table

9. FILTERS:
   - Apply WHERE before JOIN when possible
   - Use ILIKE for case-insensitive text matching
   - Only filter on columns that exist in schemas

10. OUTPUT:
   - Generate ONLY the SELECT statement
   - Do NOT include semicolon
   - Use clear aliases for aggregated columns

VALIDATION CHECKLIST (think through this):
□ Are all table names from the "Available Tables" list?
□ Are all column names explicitly shown in the schemas?
□ Do the JOIN conditions use actual foreign key relationships?
□ Is this the simplest query that answers the question?

Return ONLY the SQL query, no explanations."""

    try:
        result = call_azure_llm_with_retry(prompt, max_tokens=2000, temperature=0.0)
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
    """Analyze SQL results by examining BOTH the SQL pattern AND result structure."""
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
    """Determine how many items to include in narrative based on result count."""
    if total_results <= 100:
        return (total_results, total_results, "all_items")
    else:
        return (100, 50, "top_100_plus_summary")


def call_llm_for_narrative(prompt):
    """Call LLM to generate narrative from structured data"""
    try:
        result = call_azure_llm_with_retry(prompt, max_tokens=2000, temperature=0.1)
        narrative = result['choices'][0]['message']['content'].strip()
        return narrative
    except Exception as e:
        return f"Unable to generate narrative: {str(e)}"



def generate_narrative(user_query, analysis, sql_query):
    """Generate natural language narrative using LLM."""
    query_type = analysis["query_type"]
    total_results = analysis["total_results"]
    
    if query_type == "empty":
        return "No results were found matching your query."
    
    if query_type == "simple_count":
        count_val = analysis["summary"]["total_count"]
        count_col = analysis["summary"]["count_column"]
        
        prompt = f"""Generate a clear, factual answer to this query.

User Query: "{user_query}"

Result: The {count_col} is {count_val:,}

Generate a natural language answer using this exact number. Do not calculate anything.
Provide context about what was counted based on the user's query. Keep it 2-3 sentences."""
        
        return call_llm_for_narrative(prompt)
    
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
        
        query_lower = user_query.lower()
        is_top_n_query = any(word in query_lower for word in ['top', 'highest', 'most', 'largest'])
        is_distribution = any(word in query_lower for word in ['distribution', 'grouped by', 'breakdown'])
        
        prompt = f"""Generate a comprehensive natural language answer based on pre-calculated results.

User Query: "{user_query}"

Query Intent: {"Top N ranking" if is_top_n_query else "Distribution analysis" if is_distribution else "Aggregation"}

Analysis Results:
- Total {agg_col.replace('_', ' ')}: {total_agg:,}
- Number of distinct categories: {total_results}
- Grouped by: {', '.join(grouping_cols)}

Top {narrative_limit} {"items" if len(grouping_cols) == 1 else "combinations"} by {agg_col.replace('_', ' ')} (sorted descending):
{chr(10).join(items_text)}
"""
        
        if total_results > narrative_limit:
            remaining = total_results - narrative_limit
            prompt += f"\nAdditional info: {remaining} more {'category' if remaining == 1 else 'categories'} exist beyond the top {narrative_limit}."
        
        prompt += """

Instructions for generating the narrative:
1. Use the EXACT numbers provided above - do not calculate, round, or estimate
2. Start with a summary sentence about the overall findings
3. Explicitly mention the top items with their exact counts and percentages
4. If multiple grouping columns exist, mention the combinations clearly
5. If additional categories exist, state how many
6. Provide context based on the user's query intent
7. Do not add interpretations, recommendations, or assumptions
8. Keep it factual, clear, and informative (3-5 sentences)
9. Use natural language, not bullet points"""
        
        return call_llm_for_narrative(prompt)
    
    if query_type == "simple_select":
        columns = analysis["summary"]["columns"]
        
        prompt = f"""Generate a clear answer to this query.

User Query: "{user_query}"

Result: Found {total_results} matching records with columns: {', '.join(columns)}

Generate a concise natural language answer (2-3 sentences) stating what was found and providing context."""
        
        return call_llm_for_narrative(prompt)
    
    return "Results retrieved successfully."


def is_safe_sql(sql_query):
    """Check if SQL query is safe to execute (read-only)"""
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
            if pattern == r'\bCREATE\s+' and ('CREATED' in sql_upper or 'CREATE' in sql_query):
                if re.search(r'CREATE\s+(TABLE|INDEX|VIEW|DATABASE)', sql_upper):
                    return False, f"Query contains forbidden keyword: CREATE"
                continue
            elif pattern == r'\bUPDATE\s+' and ('UPDATED' in sql_upper or 'UPDATE' in sql_query):
                if re.search(r'UPDATE\s+\w+\s+SET', sql_upper):
                    return False, f"Query contains forbidden keyword: UPDATE"
                continue
            else:
                keyword = pattern.replace(r'\b', '').replace(r'\s+', '')
                return False, f"Query contains forbidden keyword: {keyword}"
    
    return True, "Query is safe"


def execute_generated_sql(generated_sql, max_rows=1000):
    """Execute the generated SQL safely with validation and row limit"""
    
    is_safe, message = is_safe_sql(generated_sql)
    if not is_safe:
        raise Exception(f"Unsafe SQL: {message}")
    
    conn = get_db_connection()
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
    """Paginate results with next/previous support"""
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


# ROUTES

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend HTML"""
    return send_from_directory(".", "old.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/login", methods=["POST"])
def login_route():
    """
    Accepts JSON:
      { "conn_str": "...", "azure_endpoint": "...", "azure_key": "..." }
    Validates DB connection and does a lightweight Azure key check.
    Stores values in session on success.
    """
    data = request.get_json() or {}
    conn_str = data.get("conn_str")
    azure_endpoint = data.get("azure_endpoint")
    azure_key = data.get("azure_key")

    if not all([conn_str, azure_endpoint, azure_key]):
        return jsonify({"error": "conn_str, azure_endpoint and azure_key are required"}), 400

    # Validate Postgres connection
    try:
        test_conn = psycopg2.connect(conn_str, connect_timeout=5)
        test_conn.close()
    except Exception as e:
        return jsonify({"error": "Postgres connection failed", "details": str(e)}), 400

    # Lightweight Azure endpoint/key check
    try:
        headers = {"Content-Type": "application/json", "api-key": azure_key}
        payload = {
            "messages": [{"role": "system", "content": "health check"}],
            "max_tokens": 1,
            "temperature": 0.0
        }
        resp = requests.post(azure_endpoint, headers=headers, json=payload, timeout=8)
        
        if resp.status_code >= 500:
            return jsonify({"error": "Azure endpoint error", "status": resp.status_code, "body": resp.text[:200]}), 400
        if resp.status_code == 401:
            return jsonify({"error": "Azure authentication failed (invalid key)"}), 400
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Azure endpoint check failed", "details": str(e)}), 400

    # Store in session (lowercase 'session', not 'Session')
    session["conn_str"] = conn_str
    session["azure_endpoint"] = azure_endpoint
    session["azure_key"] = azure_key

    return jsonify({"message": "Credentials stored in session"}), 200


@app.route("/table_structure", methods=["GET"])
def table_structure_route():
    """Get table metadata and LLM-generated insights"""
    schema = request.args.get("schema", "public")
    table = request.args.get("table")

    if not table:
        return jsonify({"error": "table parameter is required"}), 400

    try:
        metadata = get_table_structure_and_count(schema, table)
        llm_insights = call_llm_with_metadata(table, metadata)
        
        return jsonify({
            "table": table,
            "schema": schema,
            "metadata": metadata,
            "insights": llm_insights
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query_route():
    """Convert natural language query to SQL, execute, and generate narrative response"""
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

    try:
        if not SCHEMA_REGISTRY["loaded"]:
            load_schema_registry(schema)
        
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
                metadata = get_table_structure_and_count(schema, table_name, include_examples=True)
                insights = None
                
                if use_insights:
                    insights = call_llm_with_metadata(table_name, metadata)
                
                METADATA_CACHE[cache_key] = {
                    "metadata": metadata,
                    "insights": insights
                }
                
                all_metadata[table_name] = metadata
                all_insights[table_name] = insights
        
        generated_sql = generate_multi_table_sql(
            schema, table_names, user_query, 
            all_metadata, all_insights
        )
        
        response = {
            "user_query": user_query,
            "generated_sql": generated_sql,
            "tables_used": table_names,
            "cache_used": any(f"{schema}.{t}" in METADATA_CACHE for t in table_names) and not force_refresh
        }
        
        if execute:
            try:
                result = execute_generated_sql(generated_sql)
                raw_results = result["rows"]
                execution_time = result["execution_time_ms"]
                
                analysis = analyze_query_results(generated_sql, raw_results)
                
                if analysis["query_type"] == "aggregation_grouped":
                    processed_results = analysis["summary"]["enriched_results"]
                else:
                    processed_results = raw_results
                
                narrative = generate_narrative(user_query, analysis, generated_sql)
                
                _, preview_size, scope = determine_narrative_scope(analysis["total_results"])
                
                pagination = paginate_results(processed_results, page, preview_size)
                
                response.update({
                    "narrative": {
                        "answer": narrative,
                        "scope": scope
                    },
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
    app.run(host="0.0.0.0", port=8080)