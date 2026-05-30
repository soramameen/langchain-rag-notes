import argparse
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_markdown_documents(note_dirs: list[str]) -> list[Document]:
    documents: list[Document] = []

    for note_dir in note_dirs:
        loader = DirectoryLoader(
            note_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents.extend(loader.load())

    return documents


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain(note_dirs: list[str]):
    documents = load_markdown_documents(note_dirs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に答えてください。

文脈:
{context}

質問: {question}
""")

    model = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    return (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | prompt
        | model
        | StrOutputParser()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Markdown notes RAG example with LangChain and Chroma."
    )
    parser.add_argument(
        "--notes-dir",
        action="append",
        required=True,
        help="Markdown notes directory. Can be specified multiple times.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask against your notes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chain = build_chain(args.notes_dir)
    print(chain.invoke(args.question))
