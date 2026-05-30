# LangChain RAG Notes

LangChainでローカルMarkdownノートを検索するRAG実装集です。

最初はシンプルなRAGから始めて、HyDE、Multi Query、RRF、BM25、Web検索との組み合わせまで段階的に試せます。

## 全体像

```mermaid
flowchart LR
    Q[質問] --> R[Retriever]
    N[Markdown Notes] --> L[DirectoryLoader]
    L --> S[Text Splitter]
    S --> E[OpenAI Embeddings]
    E --> C[(Chroma)]
    C --> R
    R --> P[Prompt]
    P --> M[Chat Model]
    M --> A[回答]
```

基本は、ローカルのMarkdownノートをチャンク化してChromaに入れ、質問に近いチャンクを検索してLLMに渡す流れです。

## セットアップ

```bash
uv sync
export OPENAI_API_KEY="..."
```

任意で利用するモデルを変更できます。

```bash
export OPENAI_CHAT_MODEL="gpt-4o-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

Web検索を使う例では Brave Search API キーも必要です。

```bash
export BRAVE_SEARCH_API_KEY="..."
```

## 使い方

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

## Examples

### 1. Simple RAG

```bash
uv run python examples/simple_notes_rag.py \
  --notes-dir /path/to/notes \
  --question "私の学びを教えて"
```

基本形です。

```mermaid
flowchart LR
    Q[質問] --> V[Chroma Retriever]
    V --> D[関連ノート]
    D --> P[Prompt]
    P --> LLM[LLM]
    LLM --> A[回答]
```

### 2. HyDE

```bash
uv run python examples/hyde.py \
  --notes-dir /path/to/notes \
  --question "私の学びを教えて"
```

質問から仮の回答文を生成し、その文章で検索します。
抽象的な質問を検索しやすい意味の塊に変換する手法です。

```mermaid
flowchart LR
    Q[質問] --> H[LLMで仮回答を生成]
    H --> V[仮回答でベクトル検索]
    V --> D[関連ノート]
    D --> A[回答生成]
```

### 3. Multi Query + RRF

```bash
uv run python examples/multi_query.py \
  --notes-dir /path/to/notes \
  --question "私の学びを教えて"
```

質問から複数の検索クエリを生成し、それぞれの検索結果をRRFで統合します。

```mermaid
flowchart LR
    Q[質問] --> G[検索クエリを3つ生成]
    G --> Q1[Query 1]
    G --> Q2[Query 2]
    G --> Q3[Query 3]
    Q1 --> R1[検索結果]
    Q2 --> R2[検索結果]
    Q3 --> R3[検索結果]
    R1 --> F[RRFで統合]
    R2 --> F
    R3 --> F
    F --> A[回答生成]
```

### 4. BM25 + Vector Hybrid

```bash
uv run python examples/hybrid_bm25_vector.py \
  --notes-dir /path/to/notes \
  --question "LangChainとRAGの学びを教えて"
```

古典的なキーワード検索であるBM25と、Embeddingによる意味検索を組み合わせます。

```mermaid
flowchart LR
    Q[質問] --> B[BM25検索]
    Q --> V[ベクトル検索]
    B --> F[RRFで統合]
    V --> F
    F --> A[回答生成]
```

### 5. Router: Notes or Web

```bash
uv run python examples/router_notes_web.py \
  --notes-dir /path/to/notes \
  --question "LangChainの最新情報を教えて"
```

LLMが質問内容に応じて、ノート検索かWeb検索かを選びます。

```mermaid
flowchart LR
    Q[質問] --> Router[LLM Router]
    Router -->|自分のこと| Notes[ノート検索]
    Router -->|一般知識・最新情報| Web[Brave Web検索]
    Notes --> A[回答生成]
    Web --> A
```

### 6. Notes + Web Hybrid

```bash
uv run python examples/hybrid_notes_web.py \
  --notes-dir /path/to/notes \
  --question "最近のLangChainの情報も踏まえて、私の学びの方向性を考えて"
```

ノートの意味検索とBrave Web検索を両方実行し、RRFで統合します。

```mermaid
flowchart LR
    Q[質問] --> Notes[ノートのベクトル検索]
    Q --> Web[Brave Web検索]
    Notes --> F[RRFで統合]
    Web --> F
    F --> A[回答生成]
```

## 共通オプション

チャンクサイズは環境変数で変更できます。

```bash
export CHUNK_SIZE=500
export CHUNK_OVERLAP=100
```

## 注意

- `.env` や `.envrc` などAPIキーを含むファイルはコミットしないでください。
- Chromaの永続化DBには個人ノート由来の情報が含まれる可能性があります。公開repoには入れないでください。
