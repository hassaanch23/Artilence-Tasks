"""The RAG chain: retrieve relevant chunks, then answer grounded ONLY in them.

RAG = Retrieval-Augmented Generation. Instead of hoping the LLM memorised your
PDF, we (1) retrieve the most relevant chunks from FAISS and (2) put them into the
prompt as context. The model answers from that context -- so answers are grounded
in your document, citable, and far less prone to hallucination.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src import config
from src.llm import get_llm

SYSTEM = (
    "You are a PDF search assistant. Answer the question using ONLY the context "
    "below. If the answer is not in the context, say you don't know -- do not make "
    "anything up. Be concise, and cite the page numbers you used."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM + "\n\nContext:\n{context}"),
        ("human", "{question}"),
    ]
)


def format_docs(docs) -> str:
    """Turn retrieved Documents into one context string, each tagged with its page."""
    blocks = []
    for d in docs:
        page = d.metadata.get("page")
        tag = f"[page {page + 1}]" if isinstance(page, int) else "[page ?]"
        blocks.append(f"{tag} {d.page_content}")
    return "\n\n".join(blocks)


def build_rag_chain(index, llm=None):
    """LCEL chain (the Phase 2 pipe): retriever -> prompt -> model -> string.

    Reads as: fill {context} from the retriever and {question} from the input,
    render the prompt, call the model, parse to plain text.
    """
    retriever = index.as_retriever(search_kwargs={"k": config.TOP_K})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | (llm or get_llm())
        | StrOutputParser()
    )


def answer_with_sources(index, question: str, llm=None):
    """Return (answer, retrieved_docs) so a UI can show the source passages.

    Same prompt/model as build_rag_chain, but we retrieve explicitly so the caller
    also gets the Documents (for page citations), which the pure chain hides.
    """
    docs = index.as_retriever(search_kwargs={"k": config.TOP_K}).invoke(question)
    chain = PROMPT | (llm or get_llm()) | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    return answer, docs


if __name__ == "__main__":
    from src.vectorstore import index_exists, load_index

    if not index_exists():
        print("No index found. Run:  python -m src.ingest data/sample.pdf")
    else:
        index = load_index()
        for q in [
            "What is Project Nimbus and when was it launched?",
            "What is the capital of France?",  # NOT in the PDF -> should say it doesn't know
        ]:
            ans, docs = answer_with_sources(index, q)
            print(f"\nQ: {q}\nA: {ans}\n   (sources: {[d.metadata.get('page') for d in docs]})")
