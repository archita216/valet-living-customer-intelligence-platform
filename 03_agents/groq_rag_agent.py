import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

print("[Initializing Valet Living RAG Agent...]", flush=True)

# Load environment variables from .env file
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
print(f"[Looking for .env at: {dotenv_path}]", flush=True)
print(f"[.env exists: {dotenv_path.exists()}]", flush=True)
load_dotenv(dotenv_path=dotenv_path)

# Try direct file read if env var not loaded
if not os.getenv("GROQ_API_KEY"):
    print("[Env var empty, reading .env file directly...]", flush=True)
    try:
        with open(dotenv_path) as f:
            content = f.read()
            print(f"[.env content length: {len(content)}]", flush=True)
            for line in content.split('\n'):
                line = line.strip()
                print(f"[Reading line: {line[:50]}...]", flush=True)
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1]
                    print(f"[Found key, length: {len(key)}]", flush=True)
                    os.environ["GROQ_API_KEY"] = key
                    print(f"[Set env var, checking: {bool(os.getenv('GROQ_API_KEY'))}]", flush=True)
                    break
    except Exception as e:
        print(f"[Error reading .env: {e}]", flush=True)

print(f"[GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}]", flush=True)
if os.getenv("GROQ_API_KEY"):
    print(f"[API Key value (first 20 chars): {os.getenv('GROQ_API_KEY')[:20]}...]", flush=True)
print("[✓ Environment loaded]", flush=True)

# Configuration for RAG pipeline
VECTOR_DB_DIR = Path(__file__).resolve().parent.parent / "02_rag" / "valet_vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"


def load_embeddings() -> HuggingFaceEmbeddings:
    # Initialize HuggingFace embeddings model for semantic search
    print("[Loading embeddings model...]", flush=True)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    # Load pre-built FAISS index from disk with proper deserialization
    print("[Loading FAISS index... (this may take a minute)]", flush=True)
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {VECTOR_DB_DIR}. "
            "Run build_vectordb.py first."
        )
    return FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


def build_groq_llm() -> ChatGroq:
    # Initialize ChatGroq LLM with API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in .env file.")
    return ChatGroq(
        api_key=api_key,
        model_name=GROQ_MODEL_NAME
    )


# Prompt template for RAG
RAG_PROMPT = ChatPromptTemplate.from_template(
    """Use the following context to answer the question. If you cannot find the answer in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
)


_cached_chain = None


def build_rag_chain():
    # Load embeddings and vector store for similarity search
    print("[Initializing RAG chain...]", flush=True)
    embeddings = load_embeddings()
    vector_store = load_vector_store(embeddings)
    print("[Creating retriever...]", flush=True)
    # Create retriever that returns top 5 most relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # Initialize Groq LLM for answer generation
    llm = build_groq_llm()
    print("[Building chain...]", flush=True)
    
    # Build RAG chain combining retriever + LLM for question-answering
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
    )
    print("[✓ RAG chain ready!]\n", flush=True)
    return chain


def get_rag_chain():
    # Cache RAG chain to avoid reloading embeddings and model on each query
    global _cached_chain
    if _cached_chain is None:
        _cached_chain = build_rag_chain()
    return _cached_chain


def rag_agent(question: str) -> str:
    # Execute RAG query and return LLM-generated answer
    chain = get_rag_chain()
    result = chain.invoke(question)
    return result.content if hasattr(result, 'content') else str(result)


def main() -> None:
    # Example questions to demonstrate RAG capabilities
    sample_questions = [
        "What are the most common complaints?",
        "Which properties have high churn risk due to missed pickups?",
        "How can customer service improve satisfaction for high-risk properties?",
    ]

    import argparse

    # Parse command-line arguments for custom questions
    parser = argparse.ArgumentParser(description="Query the Valet Living RAG agent with questions about complaints and services.")
    parser.add_argument("question", nargs="?", help="Question to ask the RAG agent")
    args = parser.parse_args()

    if args.question:
        # Answer user-provided question
        print(f"\nQuestion: {args.question}")
        try:
            answer = rag_agent(args.question)
            print(f"Answer:\n{answer}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")
    else:
        # Run sample questions if no custom question provided
        for question in sample_questions:
            print(f"\nQuestion: {question}")
            try:
                answer = rag_agent(question)
                print(f"Answer:\n{answer}")
            except Exception as exc:
                print(f"Error: {exc}")
            print("-" * 80)


if __name__ == "__main__":
    main()
