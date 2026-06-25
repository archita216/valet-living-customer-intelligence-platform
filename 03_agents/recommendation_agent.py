"""
recommendation_agent.py
-----------------------
Valet Living Recommendation Agent
- Takes SQL findings + RAG complaint context
- Uses Groq LLM to generate strategic recommendations
- Acts as the final node in the LangGraph workflow
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ── Load environment variables ──
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_MODEL = "llama-3.1-8b-instant"

# ── Prompt template ──
RECOMMENDATION_PROMPT = ChatPromptTemplate.from_template("""
You are a senior customer retention strategist for Valet Living, 
a doorstep trash collection and amenity services company.

You have been given:
1. SQL data findings from the database
2. Complaint context retrieved from historical records

Your job is to analyze both and generate a clear, actionable report.

Question Asked:
{question}

SQL Data Findings:
{sql_result}

Complaint Context from Historical Records:
{rag_result}

Generate a structured response with exactly these 4 sections:

KEY FINDINGS:
(Summarize the most important insights from the SQL data and complaint context)

ROOT CAUSE:
(Identify the underlying reasons behind the problems)

RISK ASSESSMENT:
(Assess the business risk — revenue at risk, churn probability, urgency)

RECOMMENDED ACTIONS:
(List 4-5 specific, actionable steps Valet Living should take immediately)

Be specific, concise, and business-focused.
""")


def recommendation_agent(question: str, sql_result: str, rag_result: str) -> str:
    """
    Generate strategic recommendations by combining SQL data and RAG context.
    
    Args:
        question:   Original user question
        sql_result: Structured data answer from SQL agent
        rag_result: Complaint context from RAG agent
    Returns:
        Formatted recommendation report as string
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in .env file.")

    llm   = ChatGroq(api_key=api_key, model_name=GROQ_MODEL)
    chain = RECOMMENDATION_PROMPT | llm

    result = chain.invoke({
        "question":   question,
        "sql_result": sql_result,
        "rag_result": rag_result
    })

    return result.content if hasattr(result, "content") else str(result)


def main():
    """Test recommendation agent with sample inputs."""

    question = "Which properties are most at risk of churn and how can we improve retention?"

    sql_result = """
    Top high churn risk properties:
    1. Silk Road Residences | Complaints: 23 | CSAT: 1.74 | Missed Pickups: 10 | Contract: $49,985
    2. Cedar Springs Housing | Complaints: 22 | CSAT: 2.23 | Missed Pickups: 9 | Contract: $38,200
    3. Pine Creek Complex | Complaints: 22 | CSAT: 2.5 | Missed Pickups: 8 | Contract: $47,055
    4. Riverside Court | Complaints: 23 | CSAT: 2.3 | Missed Pickups: 8 | Contract: $49,983
    5. MetroLoft Residences | Complaints: 20 | CSAT: 1.23 | Missed Pickups: 10 | Contract: $49,994
    """

    rag_result = """
    Most common complaints from historical records:
    - Overall performance has degraded in the last few days
    - Missed doorstep pickups recurring weekly
    - Poor communication from service team
    - Billing issues not resolved after multiple follow-ups
    - Agent responses closed without resolution
    """

    print("Generating recommendations...\n")
    print("=" * 60)
    result = recommendation_agent(question, sql_result, rag_result)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()