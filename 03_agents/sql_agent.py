"""
Valet Living SQL Agent with LangFuse Tracing:

- Connects to Azure SQL using pyodbc
- Maps natural language questions to SQL queries
- Queries gold tables and returns formatted answers
- Used as the SQL node in the LangGraph multi-agent system
"""

import os
import pyodbc
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

from langfuse import observe

# ── Database connection config ──
SQL_SERVER   = os.getenv("SQL_SERVER")
SQL_DATABASE = os.getenv("SQL_DATABASE")
SQL_USERNAME = os.getenv("SQL_USERNAME")
SQL_PASSWORD = os.getenv("SQL_PASSWORD")
SQL_DRIVER   = "ODBC Driver 18 for SQL Server"


SUPPORTED_SQL_PATTERNS = {
    "high churn risk": ["high churn", "churn risk", "at risk of churn"],
    "low csat": ["worst csat", "low csat", "satisfaction"],
    "missed pickups": ["missed pickup", "missed pickups"],
    "contract value at risk": ["contract", "revenue at risk"],
    "service performance": ["service performance", "service category"],
    "property health": ["property health", "worst property"],
    "renewal timing": ["renewal", "renewing soon"],
}


def can_handle_question(question: str) -> bool:
    """Return True when the question matches one of the predefined SQL intents."""
    question_lower = question.lower()
    return any(
        any(pattern in question_lower for pattern in patterns)
        for patterns in SUPPORTED_SQL_PATTERNS.values()
    )


def get_connection():
    """Create and return a connection to Azure SQL."""
    conn_str = (
        f"DRIVER={{{SQL_DRIVER}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};"
        f"PWD={SQL_PASSWORD};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )
    return pyodbc.connect(conn_str)


def run_query(query: str) -> list:
    """
    Execute a SQL query and return results as a list of dicts.
    Opens connection, runs query, closes connection cleanly.
    """
    conn   = None
    cursor = None
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows    = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        raise RuntimeError(f"SQL query failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Langfuse context for observability
@observe(name="sql_agent")

def sql_agent(question: str) -> str:
    """
    Maps a natural language question to a SQL query,
    runs it against Azure SQL, and returns a formatted answer.
    
    Args:
        question: Natural language question about Valet Living data
    Returns:
        Formatted string answer with actual data from Azure SQL
    """
    question_lower = question.lower()

    if not can_handle_question(question):
        return (
            "I do not have a predefined SQL mapping for that question yet. "
            "Try asking about churn risk, CSAT, missed pickups, contract value, "
            "service performance, property health, or renewal timing."
        )

    # ── Route question to appropriate SQL query ──

    if "high churn" in question_lower or "churn risk" in question_lower or "at risk of churn" in question_lower:
        query = """
            SELECT TOP 10
                property_name,
                churn_risk_label,
                complaint_count,
                avg_csat,
                missed_pickups_30d,
                contract_value
            FROM gold_churn_risk
            WHERE churn_risk_label = 'high'
            ORDER BY complaint_count DESC
        """
        label = "Top 10 high churn risk properties"
        def format_row(r):
            return (
                f"Property: {r['property_name']} | "
                f"Complaints: {r['complaint_count']} | "
                f"CSAT: {r['avg_csat']} | "
                f"Missed Pickups: {r['missed_pickups_30d']} | "
                f"Contract: ${r['contract_value']:,.0f}"
            )

    elif "worst csat" in question_lower or "low csat" in question_lower or "satisfaction" in question_lower:
        query = """
            SELECT TOP 10
                property_name,
                avg_csat,
                complaint_count,
                churn_risk_label
            FROM gold_churn_risk
            ORDER BY avg_csat ASC
        """
        label = "Top 10 properties with lowest CSAT scores"
        def format_row(r):
            return (
                f"Property: {r['property_name']} | "
                f"CSAT: {r['avg_csat']} | "
                f"Complaints: {r['complaint_count']} | "
                f"Risk: {r['churn_risk_label']}"
            )

    elif "missed pickup" in question_lower:
        query = """
            SELECT TOP 10
                property_name,
                missed_pickups_30d,
                churn_risk_label,
                avg_csat,
                complaint_count
            FROM gold_churn_risk
            ORDER BY missed_pickups_30d DESC
        """
        label = "Top 10 properties with most missed pickups"
        def format_row(r):
            return (
                f"Property: {r['property_name']} | "
                f"Missed Pickups: {r['missed_pickups_30d']} | "
                f"Risk: {r['churn_risk_label']} | "
                f"CSAT: {r['avg_csat']}"
            )

    elif "contract" in question_lower or "revenue at risk" in question_lower:
        query = """
            SELECT TOP 10
                property_name,
                contract_value,
                churn_risk_label,
                avg_csat,
                renewal_date
            FROM gold_churn_risk
            WHERE churn_risk_label = 'high'
            ORDER BY contract_value DESC
        """
        label = "Top 10 high risk properties by contract value"
        def format_row(r):
            return (
                f"Property: {r['property_name']} | "
                f"Contract: ${r['contract_value']:,.0f} | "
                f"CSAT: {r['avg_csat']} | "
                f"Renewal: {r['renewal_date']}"
            )

    elif "service performance" in question_lower or "service category" in question_lower:
        query = """
            SELECT
                service_category,
                SUM(total_complaints)   AS total_complaints,
                AVG(avg_csat)           AS avg_csat,
                AVG(avg_resolution_time)AS avg_resolution_time,
                SUM(total_missed_pickups) AS total_missed_pickups
            FROM gold_service_performance
            GROUP BY service_category
            ORDER BY total_complaints DESC
        """
        label = "Service performance by category"
        def format_row(r):
            return (
                f"Category: {r['service_category']} | "
                f"Complaints: {r['total_complaints']} | "
                f"CSAT: {round(r['avg_csat'], 2)} | "
                f"Avg Resolution: {round(r['avg_resolution_time'], 1)}h"
            )

    elif "property health" in question_lower or "worst property" in question_lower:
        query = """
            SELECT TOP 10
                property_name,
                property_type,
                SUM(total_complaints)     AS total_complaints,
                AVG(avg_csat)             AS avg_csat,
                SUM(total_missed_pickups) AS total_missed_pickups
            FROM gold_property_health
            GROUP BY property_name, property_type
            ORDER BY total_complaints DESC
        """
        label = "Top 10 worst performing properties by health metrics"
        def format_row(r):
            return (
                f"Property: {r['property_name']} | "
                f"Type: {r['property_type']} | "
                f"Complaints: {r['total_complaints']} | "
                f"CSAT: {round(r['avg_csat'], 2)} | "
                f"Missed Pickups: {r['total_missed_pickups']}"
            )

    elif "renewal" in question_lower or "renewing soon" in question_lower:
        query = """
            SELECT TOP 10
                property_name,
                renewal_date,
                churn_risk_label,
                contract_value,
                avg_csat
            FROM gold_churn_risk
            WHERE renewal_date <= DATEADD(day, 30, GETDATE())
            ORDER BY renewal_date ASC
        """
        label = "Properties renewing within next 30 days"
        def format_row(r):
            return (
                f"Property: {r['property_name']} | "
                f"Renewal: {r['renewal_date']} | "
                f"Risk: {r['churn_risk_label']} | "
                f"Contract: ${r['contract_value']:,.0f}"
            )

    # ── Run query and format results ──
    try:
        rows = run_query(query)

        if not rows:
            return f"No data found for: {label}"

        formatted = [f"{i+1}. {format_row(r)}" for i, r in enumerate(rows)]
        return f"{label}:\n\n" + "\n".join(formatted)

    except RuntimeError as e:
        return f"Database error: {e}"

def main():
    """Test the SQL agent with sample questions."""
    test_questions = [
        "Which properties have high churn risk?",
        "Which properties have the worst CSAT scores?",
        "Show me properties with most missed pickups",
        "What is the contract value at risk?",
        "Which service categories have most complaints?",
        "Which properties have the worst property health?",
        "Which properties are renewing soon?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        answer = sql_agent(question)
        print(answer)
        print()


if __name__ == "__main__":
    main()