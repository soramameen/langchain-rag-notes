import os
from enum import Enum
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import BraveSearchLoader
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag.core import format_docs, reciprocal_rank_fusion

ANSWER_PROMPT = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。

文脈:
{context}

質問: {question}
""")


def _build_base_chain(model: ChatOpenAI):
    return ANSWER_PROMPT | model | StrOutputParser()


# ---------------------------------------------------------------------------
# 1. Simple
# ---------------------------------------------------------------------------
def build_simple_chain(db: Chroma, model: ChatOpenAI, **_):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | _build_base_chain(model)
    )


# ---------------------------------------------------------------------------
# 2. HyDE
# ---------------------------------------------------------------------------
def build_hyde_chain(db: Chroma, model: ChatOpenAI, **_):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    hypothetical_prompt = ChatPromptTemplate.from_template("""\
あなたはユーザー本人の過去ノートを検索するための、仮の回答文を作ります。
質問に直接答えるのではなく、その質問に関連しそうな概念・出来事・感情・学習テーマを含んだ文章を作ってください。
ユーザーに追加質問はしないでください。
検索に使うため、具体的なキーワードを多めに含めてください。

質問: {question}
""")
    hypothetical_chain = hypothetical_prompt | model | StrOutputParser()
    return (
        {
            "question": RunnablePassthrough(),
            "context": hypothetical_chain | retriever | format_docs,
        }
        | _build_base_chain(model)
    )


# ---------------------------------------------------------------------------
# 3. Multi Query + RRF
# ---------------------------------------------------------------------------
class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="Generated search queries")


def build_multi_query_chain(db: Chroma, model: ChatOpenAI, **_):
    retriever = db.as_retriever(search_kwargs={"k": 5})
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
    return (
        {
            "question": RunnablePassthrough(),
            "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion | format_docs,
        }
        | _build_base_chain(model)
    )


# ---------------------------------------------------------------------------
# 4. BM25 + Vector Hybrid
# ---------------------------------------------------------------------------
def build_hybrid_bm25_vector_chain(db: Chroma, chunks: list[Document], model: ChatOpenAI, **_):
    vector_retriever = db.as_retriever(search_kwargs={"k": 5})
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
    return (
        {"question": RunnablePassthrough(), "context": hybrid_retriever}
        | _build_base_chain(model)
    )


# ---------------------------------------------------------------------------
# 5. Router: Notes or Web
# ---------------------------------------------------------------------------
class Route(str, Enum):
    notes = "notes"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


def _brave_search_retriever(query: str) -> list[Document]:
    docs = BraveSearchLoader(
        query=query,
        api_key=os.environ["BRAVE_SEARCH_API_KEY"],
        search_kwargs={"count": 5},
    ).load()
    for doc in docs:
        doc.metadata["source_type"] = "web"
    return docs


def build_router_chain(db: Chroma, model: ChatOpenAI, **_):
    notes_retriever = db.as_retriever(search_kwargs={"k": 5})
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
            return format_docs(_brave_search_retriever(question))
        raise ValueError(f"Unknown route: {route}")

    return (
        {"question": RunnablePassthrough(), "route": route_chain}
        | RunnablePassthrough.assign(context=routed_retriever)
        | _build_base_chain(model)
    )


# ---------------------------------------------------------------------------
# 6. Notes + Web Hybrid
# ---------------------------------------------------------------------------
def build_hybrid_notes_web_chain(db: Chroma, model: ChatOpenAI, **_):
    notes_retriever = db.as_retriever(search_kwargs={"k": 5})
    hybrid_retriever = (
        RunnableParallel(
            {
                "notes_documents": notes_retriever,
                "web_documents": RunnablePassthrough() | _brave_search_retriever,
            }
        )
        | (lambda x: [x["notes_documents"], x["web_documents"]])
        | reciprocal_rank_fusion
        | format_docs
    )
    return (
        {"question": RunnablePassthrough(), "context": hybrid_retriever}
        | _build_base_chain(model)
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_STRATEGY_BUILDERS = {
    "simple": build_simple_chain,
    "hyde": build_hyde_chain,
    "multi-query": build_multi_query_chain,
    "hybrid-bm25-vector": build_hybrid_bm25_vector_chain,
    "router": build_router_chain,
    "hybrid-notes-web": build_hybrid_notes_web_chain,
}


def list_strategies() -> list[str]:
    return list(_STRATEGY_BUILDERS.keys())


def build_chain(strategy: str, db: Chroma, chunks: list[Document] | None, model: ChatOpenAI):
    if strategy not in _STRATEGY_BUILDERS:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list_strategies()}")
    builder = _STRATEGY_BUILDERS[strategy]
    return builder(db=db, chunks=chunks, model=model)
