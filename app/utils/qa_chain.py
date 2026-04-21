from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.utils.embedder import load_vectorstore


def get_llm():
    """Initialize Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
    )


def get_retriever():
    """Load vectorstore and return retriever."""
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": settings.retriever_k}
    )


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(question: str) -> dict:
    """Ask a question and return answer with source chunks."""
    retriever = get_retriever()
    llm = get_llm()

    # Get relevant chunks
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
""")

    # Get answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Return answer + sources
    sources = [
        {
            "content": doc.page_content[:200],
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in docs
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }