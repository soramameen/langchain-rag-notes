from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from common import (
    build_chat_model,
    build_vectorstore,
    format_docs,
    load_markdown_documents,
    parse_common_args,
    split_documents,
)


def build_chain(note_dirs: list[str]):
    documents = load_markdown_documents(note_dirs)
    chunks = split_documents(documents)
    db = build_vectorstore(chunks)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。

文脈:
{context}

質問: {question}
""")

    return (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | prompt
        | build_chat_model()
        | StrOutputParser()
    )


if __name__ == "__main__":
    args = parse_common_args("Simple local Markdown notes RAG example.")
    chain = build_chain(args.notes_dir)
    print(chain.invoke(args.question))
