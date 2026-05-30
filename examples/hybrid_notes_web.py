import os

from langchain_community.document_loaders import BraveSearchLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from common import (
    build_chat_model,
    build_vectorstore,
    format_docs,
    load_markdown_documents,
    parse_common_args,
    reciprocal_rank_fusion,
    split_documents,
)


def brave_search_retriever(query: str) -> list[Document]:
    docs = BraveSearchLoader(
        query=query,
        api_key=os.environ["BRAVE_SEARCH_API_KEY"],
        search_kwargs={"count": 5},
    ).load()
    for doc in docs:
        doc.metadata["source_type"] = "web"
    return docs


def build_chain(note_dirs: list[str]):
    documents = load_markdown_documents(note_dirs)
    chunks = split_documents(documents)
    notes_retriever = build_vectorstore(chunks).as_retriever(search_kwargs={"k": 5})

    hybrid_retriever = (
        RunnableParallel(
            {
                "notes_documents": notes_retriever,
                "web_documents": RunnablePassthrough() | brave_search_retriever,
            }
        )
        | (lambda x: [x["notes_documents"], x["web_documents"]])
        | reciprocal_rank_fusion
        | format_docs
    )

    prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。
ノート由来の情報とWeb由来の情報は区別して扱ってください。

文脈:
{context}

質問: {question}
""")

    return (
        {"question": RunnablePassthrough(), "context": hybrid_retriever}
        | prompt
        | build_chat_model()
        | StrOutputParser()
    )


if __name__ == "__main__":
    args = parse_common_args("Notes vector search + Brave web search hybrid RAG example.")
    print(build_chain(args.notes_dir).invoke(args.question))
