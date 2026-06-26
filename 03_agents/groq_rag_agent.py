"""
Valet Living RAG Agent with LangFuse tracing:

- Loads the FAISS vector DB built from 100k complaint records
- Uses HuggingFace embeddings for semantic search
- Uses Groq (llama3-8b-8192) as the LLM to generate answers
- Answers natural language questions about client complaints and churn
"""


import os
import argparse
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langfuse import observe

# ── Configuration ──
VECTOR_DB_DIR       = Path(__file__).resolve().parent.parent / "02_rag" / "valet_vector_db"
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL          = "llama-3.1-8b-instant"

# ── Prompt template ──
PROMPT = ChatPromptTemplate.from_template("""
You are a Valet Living client success assistant.
Use the complaint records below to answer the question clearly and concisely.
If the answer is not in the context, say "I could not find relevant information."

Context:
{context}

Question: {question}

Answer:
""")

def load_embeddings():
    """Load the same HuggingFace model used to build the FAISS index."""
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vector_store(embeddings):
    """Load the pre-built FAISS index from disk."""
    print(f"Loading FAISS index from {VECTOR_DB_DIR}...")
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {VECTOR_DB_DIR}\n"
            "Please run build_vectordb.py first."
        )
    return FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

def load_groq_llm():
    """Initialize Groq LLM using API key from .env file."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found.\n"
            "Add it to your .env file like this:\n"
            "GROQ_API_KEY=gsk_your_key_here"
        )
    print(f"Groq API key loaded ({api_key[:8]}...)")
    return ChatGroq(api_key=api_key, model_name=GROQ_MODEL)


# ── Build RAG chain (cached so it loads only once) ──
_chain = None

def get_rag_chain():
    """
    Build the RAG pipeline:
    1. User question → HuggingFace embeddings → FAISS search → top 5 docs
    2. Top 5 docs + question → Groq LLM → natural language answer
    """
    global _chain
    if _chain is not None:
        return _chain

    embeddings   = load_embeddings()
    vector_store = load_vector_store(embeddings)
    retriever    = vector_store.as_retriever(search_kwargs={"k": 5})
    llm          = load_groq_llm()

    def format_docs(docs):
        """Join retrieved document chunks into one context string."""
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    _chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
    )
    print("RAG chain ready.\n")
    return _chain

#Langfuse decorator to observe the function
@observe(name="rag_agent") 
def ask(question: str) -> str:
    """Ask a question and get a natural language answer from the RAG agent."""
    chain  = get_rag_chain()
    result = chain.invoke(question)
    return result.content if hasattr(result, "content") else str(result)

def main():
    parser = argparse.ArgumentParser(
        description="Query the Valet Living RAG agent"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (optional — runs demo questions if not provided)"
    )
    args = parser.parse_args()

    if args.question:
        # Single question mode
        print(f"\nQuestion: {args.question}")
        print(f"Answer:\n{ask(args.question)}\n")
    else:
        # Demo mode — runs 3 sample questions
        demo_questions = [
            "What are the most common complaints from customers?",
            "Which properties have high churn risk due to missed pickups?",
            "How did agents resolve billing disputes?",
        ]
        for q in demo_questions:
            print(f"\nQuestion: {q}")
            print(f"Answer:\n{ask(q)}")
            print("-" * 80)


if __name__ == "__main__":
    main()