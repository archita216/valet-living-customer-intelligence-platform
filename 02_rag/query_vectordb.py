import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

VECTOR_DB_DIR = Path(__file__).resolve().parent / "valet_vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"


def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in .env file.")
    return ChatGroq(
        api_key=api_key,
        model_name=GROQ_MODEL_NAME
    )


_qa_chain = None


def build_retrieval_qa() -> RetrievalQA:
    embeddings = load_embeddings()
    vector_store = load_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = build_groq_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )


def get_qa_chain() -> RetrievalQA:
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = build_retrieval_qa()
    return _qa_chain


def ask(question: str) -> str:
    """Return a natural language answer for the provided question."""
    qa_chain = get_qa_chain()
    return qa_chain.run(question)


def main() -> None:
    questions = [
        "Which properties have high churn risk due to missed pickups?",
        "What are the most common complaints from customers?",
        "How did agents resolve billing disputes?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            answer = ask(question)
            print(f"Answer: {answer}")
        except Exception as exc:
            print(f"Error: {exc}")
        print("-" * 80)


if __name__ == "__main__":
    main()