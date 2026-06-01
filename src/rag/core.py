import os
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_markdown_documents(
    note_dirs: Iterable[str],
    source_type: str = "notes",
) -> list[Document]:
    documents: list[Document] = []
    for note_dir in note_dirs:
        loader = DirectoryLoader(
            note_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source_type"] = source_type
        documents.extend(loaded_docs)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def build_embeddings():
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def build_vectorstore(
    chunks: list[Document],
    persist_directory: str | None = None,
) -> Chroma:
    embeddings = build_embeddings()
    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory,
    )


def load_vectorstore(persist_directory: str) -> Chroma:
    embeddings = build_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def build_chat_model() -> ChatOpenAI:
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


def format_docs(docs: list[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs, start=1):
        source_type = doc.metadata.get("source_type", "unknown")
        source = doc.metadata.get("source") or doc.metadata.get("link") or "unknown"
        title = doc.metadata.get("title", "")
        formatted_docs.append(
            f"[{i}] source_type={source_type}\n"
            f"source={source}\n"
            f"title={title}\n"
            f"content:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_docs)


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    content_score_mapping: dict[str, float] = {}
    content_doc_mapping: dict[str, Document] = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content
            if content not in content_score_mapping:
                content_score_mapping[content] = 0.0
                content_doc_mapping[content] = doc
            content_score_mapping[content] += 1 / (rank + 1 + k)

    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content_doc_mapping[content] for content, _ in ranked]
