# Valet Living Client Churn & Service Risk Intelligence Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi%20Agent-green)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-orange)
![Azure SQL](https://img.shields.io/badge/Azure%20SQL-Database-0078D4)

## 🎯 Problem Statement

Valet Living manages thousands of properties across multiple states, and client success teams need a faster way to detect churn risk and service issues before they escalate. Traditional reporting is often reactive, making it difficult to answer questions like:

- Which properties are most at risk of churn?
- Why are certain clients unhappy?
- What service issues are recurring across accounts?
- What action should the client success team take next?

This project builds an AI-powered multi-agent system that helps teams identify service risks, understand complaint patterns, and generate retention-focused recommendations.

## 💡 Proposed Solution

A multi-agent Retrieval-Augmented Generation (RAG) system powered by LangGraph and Groq LLMs.

The solution combines:

- Structured analytics from Azure SQL
- Unstructured complaint and service-history retrieval from a FAISS vector database
- A LangGraph workflow that routes user questions to the right specialized agent
- Langfuse tracing for observability and monitoring

This creates a practical assistant for client success and retention decision-making.

## 🧠 What the Project Does

This project acts as a smart support and retention copilot for Valet Living by answering business questions through three specialized agents:

- **SQL Agent**: answers structured questions using Azure SQL data such as churn risk, CSAT, missed pickups, renewal dates, and contract value.
- **RAG Agent**: searches historical complaint context stored in a FAISS vector database to understand recurring service issues and customer pain points.
- **Recommendation Agent**: combines SQL and RAG outputs to produce clear, actionable retention recommendations.
- **LangGraph Workflow**: orchestrates the flow between agents and connects them into one end-to-end pipeline.

## 🏗️ Architecture

```text
User Question
   ↓
LangGraph Workflow
   ↓
SQL Agent  →  RAG Agent  →  Recommendation Agent
   ↓             ↓                ↓
Azure SQL   FAISS Vector DB   Groq LLM
```

## 🛠️ Tech Stack

- **Python** for the agent logic and orchestration
- **LangGraph** for multi-agent workflow orchestration
- **Langfuse** for tracing and observability
- **Groq API** for LLM-based reasoning and generation
- **FAISS** for vector search over complaint history
- **Azure SQL** for structured business data
- **python-dotenv** for environment management
- **sentence-transformers** for embeddings

## 📁 Project Structure

```text
01_data/
    silver_rag_source.csv

02_rag/
    build_vectordb.py
    valet_vector_db/
        index.faiss

03_agents/
    groq_rag_agent.py
    langgraph_workflow.py
    recommendation_agent.py
    sql_agent.py

requirements.txt
README.md
.env
```

## 🤖 Agent Responsibilities

### 1. SQL Agent
Handles structured business questions such as:
- Which properties have the highest churn risk?
- Which accounts have the lowest CSAT?
- Which properties have the most missed pickups?

It queries Azure SQL and returns business-focused insights.

### 2. RAG Agent
Handles unstructured or contextual questions such as:
- What issues have clients complained about repeatedly?
- Why is a certain property unhappy?
- What service problems are appearing in historical records?

It uses the FAISS vector database to retrieve relevant complaint context and generate a grounded answer.

### 3. Recommendation Agent
Combines insights from both the SQL and RAG agents and produces a concise action-oriented recommendation report.

### 4. LangGraph Workflow
Orchestrates the full pipeline so the system works as a single multi-agent application rather than separate tools.

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd valet-living-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root with the following values:

```env
GROQ_API_KEY=your_groq_api_key
SQL_SERVER=your_sql_server
SQL_DATABASE=your_sql_database
SQL_USERNAME=your_sql_username
SQL_PASSWORD=your_sql_password
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 5. Build the vector database
If the FAISS index is not already present, run:

```bash
python 02_rag/build_vectordb.py
```

## ▶️ Run the Project

### Run the full workflow

```bash
python 03_agents/langgraph_workflow.py
```

### Run the RAG agent directly

```bash
python 03_agents/groq_rag_agent.py
```

### Run the SQL agent directly

```bash
python 03_agents/sql_agent.py
```

### Run the recommendation agent directly

```bash
python 03_agents/recommendation_agent.py
```

## 🔍 Observability

Langfuse tracing is integrated to monitor:
- which agent executed
- how long each step took
- how prompts and outputs were generated

This makes the system easier to debug and improve.

## ✅ Summary

This project demonstrates a practical multi-agent AI solution for client success and churn-risk intelligence using:

- **LangGraph** for orchestration
- **Azure SQL** for structured data
- **FAISS + RAG** for complaint retrieval
- **Groq LLMs** for reasoning and recommendations
- **Langfuse** for observability

It is designed to help Valet Living move from reactive service management to proactive customer risk intelligence.
