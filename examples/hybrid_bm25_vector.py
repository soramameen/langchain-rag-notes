from langchain_community.retrievers.bm25 import BM25Retriever
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


def build_chain(note_dirs: list[str]):
    documents = load_markdown_documents(note_dirs)
    chunks = split_documents(documents)

    vector_retriever = build_vectorstore(chunks).as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    hybrid_retriever = (
        RunnableParallel(
            {
                "vector_documents": vector_retriever,
                "bm25_documents": bm25_retriever,
            }
        )
        | (lambda x: [x["vector_documents"], x["bm25_documents"]])
        | reciprocal_rank_fusion
        | format_docs
    )

    prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。

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
    args = parse_common_args("BM25 + vector hybrid RAG example.")
    print(build_chain(args.notes_dir).invoke(args.question))
