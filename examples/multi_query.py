from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from common import (
    build_chat_model,
    build_vectorstore,
    format_docs,
    load_markdown_documents,
    parse_common_args,
    reciprocal_rank_fusion,
    split_documents,
)


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="Generated search queries")


def build_chain(note_dirs: list[str]):
    documents = load_markdown_documents(note_dirs)
    chunks = split_documents(documents)
    retriever = build_vectorstore(chunks).as_retriever(search_kwargs={"k": 5})
    model = build_chat_model()

    query_generation_prompt = ChatPromptTemplate.from_template("""\
あなたはユーザー本人の過去ノートを検索するための検索クエリを作ります。
質問に対して、ベクターデータベースから関連する過去ノートを見つけるために、3つの異なる検索クエリを生成してください。
一般的なアドバイスではなく、ユーザー本人の過去の経験・感情・学習記録・関心・振り返りが見つかりやすい語彙を使ってください。

質問: {question}
""")
    query_generation_chain = (
        query_generation_prompt
        | model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )

    prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。

文脈:
{context}

質問: {question}
""")

    return (
        {
            "question": RunnablePassthrough(),
            "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion | format_docs,
        }
        | prompt
        | model
        | StrOutputParser()
    )


if __name__ == "__main__":
    args = parse_common_args("Multi-query RAG example for local Markdown notes.")
    print(build_chain(args.notes_dir).invoke(args.question))
