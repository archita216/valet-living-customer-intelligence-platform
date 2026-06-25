import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

load_dotenv()

# ── Step 1: Read silver data from local CSV ──
CSV_PATH = Path(__file__).resolve().parent.parent / "01_data" / "silver_rag_source.csv"
print(f"Reading silver RAG data from {CSV_PATH}...")

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found at {CSV_PATH}. "
        "Export silver table from Databricks first."
    )

df = pd.read_csv(CSV_PATH)
print("CSV shape:", df.shape)  # add this line

# Fill nulls
df["complaint_description"] = df["complaint_description"].fillna("No description provided")
df["service_response"]      = df["service_response"].fillna("No response recorded")
df["resolution_notes"]      = df["resolution_notes"].fillna("Unresolved")
df["service_category"]      = df["service_category"].fillna("General")
df["churn_risk_label"]      = df["churn_risk_label"].fillna("unknown")

print(f"Loaded {len(df)} rows")

# ── Step 2: Convert to LangChain Documents ──
print("Creating documents...")

docs = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = f"""
Service Category: {row['service_category']}
Churn Risk: {row['churn_risk_label']}

Customer Complaint:
{row['complaint_description']}

Agent Response:
{row['service_response']}

Resolution:
{row['resolution_notes']}
    """.strip()

    docs.append(Document(
        page_content=text,
        metadata={
            "property_id":      str(row["property_id"]),
            "service_category": str(row["service_category"]),
            "churn_risk":       str(row["churn_risk_label"])
        }
    ))

print(f"Created {len(docs)} documents")

# ── Step 3: Load embedding model ──
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── Step 4: Build FAISS in batches ──
print("Building FAISS index — 30-60 mins for 100k rows...")
batch_size = 500
vector_db  = None

for i in tqdm(range(0, len(docs), batch_size)):
    batch = docs[i:i + batch_size]
    if vector_db is None:
        vector_db = FAISS.from_documents(batch, embeddings)
    else:
        vector_db.add_documents(batch)

# ── Step 5: Save FAISS index ──
save_path = Path(__file__).resolve().parent / "valet_vector_db"
vector_db.save_local(str(save_path))
print(f"Vector DB saved to {save_path}")

# ── Step 6: Test retrieval ──
print("\nTesting retrieval...")
results = vector_db.similarity_search(
    "high churn risk property with missed pickups",
    k=3
)

for i, r in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(r.page_content[:300])
    print("Metadata:", r.metadata)

print("\nRAG Vector DB build complete!")