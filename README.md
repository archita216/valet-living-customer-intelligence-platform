# Valet Living Client Churn & Service Risk Intelligence Agent

![Databricks](https://img.shields.io/badge/Databricks-ETL-EA3E23)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi%20Agent-green)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-orange)
![Azure SQL](https://img.shields.io/badge/Azure%20SQL-Database-0078D4)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5)

## 🎯 Problem Statement

Valet Living manages thousands of properties across multiple states, and client success teams need a faster way to detect churn risk and service issues before they escalate. Traditional reporting is often reactive, making it difficult to answer questions like:

- Which properties are most at risk of churn?
- Why are certain clients unhappy?
- What service issues are recurring across accounts?
- What action should the client success team take next?

This project builds an AI-powered multi-agent system that helps teams identify service risks, understand complaint patterns, and generate retention-focused recommendations.

## 💡 Proposed Solution

A multi-agent Retrieval-Augmented Generation (RAG) system powered by LangGraph and Groq LLMs.

The solution begins by ingesting raw complaint and feedback data into Databricks, where it is processed through the Medallion Architecture (Bronze → Silver → Gold) using PySpark. The curated gold business tables are then published to Azure SQL Database, providing a structured analytics layer for dashboards and AI agents.

The data foundation is used in two different ways:

- The **Silver RAG table** is used to build the FAISS vector store that powers the RAG chatbot.
- The **Gold tables in Azure SQL** are used by the SQL Agent for structured analysis and reporting.

On top of this data foundation, the platform combines:

- Structured analytics from Azure SQL
- Unstructured complaint and service-history retrieval from a FAISS vector database
- A LangGraph workflow that routes user questions to the right specialized agent
- Langfuse tracing for observability and monitoring

Together, these components create an intelligent assistant that helps client success teams proactively identify service issues, detect churn risk, and recommend retention strategies.

## 🧠 What the Project Does

This project acts as a smart support and retention copilot for Valet Living by answering business questions through three specialized agents:

- **SQL Agent**: answers structured questions using Azure SQL data such as churn risk, CSAT, missed pickups, renewal dates, and contract value.
- **RAG Agent**: searches historical complaint context stored in a FAISS vector database to understand recurring service issues and customer pain points.
- **Recommendation Agent**: combines SQL and RAG outputs to produce clear, actionable retention recommendations.
- **LangGraph Workflow**: orchestrates the flow between agents and connects them into one end-to-end pipeline.
- **Streamlit Frontend**: provides an interactive interface for asking questions and viewing the results in a polished UI.

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

## 🔄 Data Engineering Pipeline

The project follows the Databricks Medallion Architecture to transform raw operational data into AI-ready business datasets.

- **Bronze Layer:** Ingests raw CSV complaint and feedback data into Databricks.
- **Silver Layer:** Cleanses, validates, standardizes, and enriches the raw records using PySpark transformations.
- **Silver RAG Table:** Produces the curated complaint context used to build the FAISS vector database for the RAG chatbot.
- **Gold Layer:** Creates business-ready analytical views, including churn risk analysis, service performance metrics, and property health summaries.
- **Azure SQL Integration:** Publishes the curated Gold tables to Azure SQL Database, which acts as the structured data source for Power BI dashboards and the SQL Agent within the multi-agent AI system.

This ETL pipeline provides a reliable and scalable data foundation for downstream analytics and AI workloads.

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

app.py

04_databricks/
    01_ Bronze Ingestion.py
    02_ Silver Transformation.py
    03_Gold Business Views.py
    04_Azure SQL Integration.py

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

If a question does not match one of the predefined SQL intents, the agent returns a clear fallback message instead of inventing a query. The Streamlit UI can then route the question to RAG or show the fallback directly.

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

## ▶️ Run the Project

### Run the full workflow

```bash
python 03_agents/langgraph_workflow.py
```

### Run the Streamlit frontend

```bash
streamlit run app.py
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

## 🐳 Containerization & CI/CD Deployment to Azure

Because local Docker Desktop installation can be restricted on managed corporate devices (e.g., due to virtualization policies), this project is structured to run **Docker container builds entirely in the cloud** via GitHub Actions, bypassing local constraints.

### ⚙️ Container Architecture & Bypassing Large Git Files
1. **Self-Contained DB Build:** The RAG Agent relies on a FAISS vector database. Since the database files are large (~196 MB total) and gitignored, the `Dockerfile` is configured to run `python 02_rag/build_vectordb.py` **during the Docker image build process**. 
2. **Resulting Benefit:** This automatically downloads the embeddings model, processes `silver_rag_source.csv` (which is committed to git), and builds the vector database inside the container image. The image is fully self-contained and starts immediately on Azure without external mounts or post-startup delays.

---

### 🛠️ Step 1: Set up Azure Resources

1. **Azure Container Registry (ACR):**
   - Go to the Azure Portal and search for **Container Registries**.
   - Create a registry (e.g., `valetregistry.azurecr.io`).
   - Under **Access Keys**, enable the **Admin user** and copy the **Username** and **Password** (you'll use these in GitHub).

2. **Azure App Service (Web App for Containers):**
   - Search for **App Services** and click **Create** -> **Web App**.
   - **Publish:** Select **Docker Container**.
   - **Operating System:** Select **Linux**.
   - **App Service Plan:** Choose your region and pricing tier.
   - **Docker options:** Select *Azure Container Registry* or *Docker Hub* as the source (can choose a placeholder image to start; GitHub Actions will overwrite it on push).

3. **Configure App Settings (Environment Variables):**
   - In your Azure App Service, go to **Configuration** (under *Settings*).
   - Add the following environment variables (which are read by `app.py`):
     - `GROQ_API_KEY`
     - `SQL_SERVER`
     - `SQL_DATABASE`
     - `SQL_USERNAME`
     - `SQL_PASSWORD`
     - `LANGFUSE_PUBLIC_KEY`
     - `LANGFUSE_SECRET_KEY`
     - `LANGFUSE_BASE_URL`
   - Click **Save** to apply the configuration.

---

### 🚀 Step 2: Configure GitHub Secrets

In your GitHub repository, go to **Settings** -> **Secrets and variables** -> **Actions** and add the following secrets:

1. `ACR_USERNAME` - The admin username for your Azure Container Registry.
2. `ACR_PASSWORD` - The admin password for your Azure Container Registry.
3. `AZURE_CREDENTIALS` - Service Principal credentials to authorize the deployment. Generate them using the Azure CLI:
   ```bash
   az ad sp create-for-rbac --name "valet-living-deployer" --role contributor --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group-name> --sdk-auth
   ```
   *Alternative:* You can use the Azure **Publish Profile** secret (`AZURE_WEBAPP_PUBLISH_PROFILE`) downloaded from the App Service overview page, and update the github action workflow `azure/webapps-deploy` step to use `publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}` instead of `creds`.

---

### 🔄 Step 3: CI/CD Deployment Flow

A GitHub Action is pre-configured in [.github/workflows/azure-container-deploy.yml](file:///C:/Users/archita.saha/OneDrive%20-%20Accenture/Desktop/valet%20living%20agent/.github/workflows/azure-container-deploy.yml).

Every time you execute a `git push` to the `main` branch:
1. **GitHub Actions** checks out the repository.
2. It logs in to your **Azure Container Registry (ACR)**.
3. It builds the Docker image, runs the vector DB build, and tags it as `latest`.
4. It pushes the completed image to your ACR.
5. It deploys the new container version to **Azure App Service**.
6. The application is updated automatically without manual intervention!

---

### 💻 Local Container Testing (Optional)

If you are running on a machine with Docker installed, you can build and run the container locally:

```bash
# Build the Docker image (takes ~5-10 minutes to run build_vectordb.py)
docker build -t valet-living-agent .

# Run the container (passing environment variables from your local .env file)
docker run --env-file .env -p 8501:8501 valet-living-agent
```
Open [http://localhost:8501](http://localhost:8501) in your browser to verify.

---

## ✅ Summary

This project demonstrates a practical multi-agent AI solution for client success and churn-risk intelligence using:

- **LangGraph** for orchestration
- **Azure SQL** for structured data
- **FAISS + RAG** for complaint retrieval
- **Groq LLMs** for reasoning and recommendations
- **Langfuse** for observability
- **Docker + GitHub Actions** for automated CI/CD container deployment to **Azure App Service**

It is designed to help Valet Living move from reactive service management to proactive customer risk intelligence.
