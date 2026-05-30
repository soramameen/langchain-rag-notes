import os
from enum import Enum
from typing import Any

from langchain_community.document_loaders import BraveSearchLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from common import (
    build_chat_model,
    build_vectorstore,
    format_docs,
    load_markdown_documents,
    parse_common_args,
    split_documents,
)


class Route(str, Enum):
    notes = "notes"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


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
    model = build_chat_model()

    route_prompt = ChatPromptTemplate.from_template("""\
質問に回答するために適切なRetrieverを選択してください。

選択肢:
- notes: ユーザー本人の過去ノート、経験、学び、考え、価値観、悩み、興味についての質問
- web: 一般知識、最新情報、外部サービス、公式ドキュメント、ニュースについての質問

質問: {question}
""")
    route_chain = route_prompt | model.with_structured_output(RouteOutput) | (lambda x: x.route)

    def routed_retriever(inp: dict[str, Any]) -> str:
        question = inp["question"]
        route = inp["route"]
        if route == Route.notes:
            return format_docs(notes_retriever.invoke(question))
        if route == Route.web:
            return format_docs(brave_search_retriever(question))
        raise ValueError(f"Unknown route: {route}")

    prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。

文脈:
{context}

質問: {question}
""")

    return (
        {"question": RunnablePassthrough(), "route": route_chain}
        | RunnablePassthrough.assign(context=routed_retriever)
        | prompt
        | model
        | StrOutputParser()
    )


if __name__ == "__main__":
    args = parse_common_args("Router RAG example for notes or Brave web search.")
    print(build_chain(args.notes_dir).invoke(args.question))
