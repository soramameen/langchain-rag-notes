# LangChain RAG Practice

LangChainでローカルMarkdownノートを検索する、シンプルなRAG実装です。

## セットアップ

```bash
uv sync
export OPENAI_API_KEY="..."
```

任意で利用するチャットモデルを変更できます。

```bash
export OPENAI_CHAT_MODEL="gpt-4o-mini"
```

## シンプルなローカルノートRAG

Markdownファイルが入ったディレクトリを指定して実行します。

```bash
uv run python examples/simple_notes_rag.py \
  --notes-dir /path/to/your/notes \
  --question "私の学びを教えて"
```

複数ディレクトリも指定できます。

```bash
uv run python examples/simple_notes_rag.py \
  --notes-dir /path/to/notes1 \
  --notes-dir /path/to/notes2 \
  --question "最近の関心を教えて"
```

## このサンプルでやっていること

1. `DirectoryLoader` でMarkdownを読み込む
2. `RecursiveCharacterTextSplitter` でチャンクに分割する
3. `OpenAIEmbeddings` でベクトル化する
4. `Chroma` に保存する
5. Retrieverで関連チャンクを検索する
6. 検索結果を文脈としてLLMに渡す

## 今後追加したいもの

学んだ内容を組み合わせた発展版も追加予定です。

- HyDE
- Multi Query
- RAG Fusion / RRF
- BM25 + ベクトル検索
- Web検索とのハイブリッド
- Retriever routing
