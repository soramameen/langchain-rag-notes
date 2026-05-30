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
    retriever = build_vectorstore(chunks).as_retriever(search_kwargs={"k": 5})
    model = build_chat_model()

    hypothetical_prompt = ChatPromptTemplate.from_template("""\
あなたはユーザー本人の過去ノートを検索するための、仮の回答文を作ります。
質問に直接答えるのではなく、その質問に関連しそうな概念・出来事・感情・学習テーマを含んだ文章を作ってください。
ユーザーに追加質問はしないでください。
検索に使うため、具体的なキーワードを多めに含めてください。

質問: {question}
""")
    hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

    prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。

文脈:
{context}

質問: {question}
""")

    return (
        {
            "question": RunnablePassthrough(),
            "context": hypothetical_chain | retriever | format_docs,
        }
        | prompt
        | model
        | StrOutputParser()
    )


if __name__ == "__main__":
    args = parse_common_args("HyDE RAG example for local Markdown notes.")
    print(build_chain(args.notes_dir).invoke(args.question))
