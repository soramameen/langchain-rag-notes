---
name: rag-notes
description: Use when the user asks about their own past notes, personal learnings, experiences, thoughts, diary entries, interests, worries, or anything that might be recorded in their local Markdown files.
---

# rag-notes

Also use when the user says things like:
- "私のノートを探して"
- "過去の学びを教えて"
- "自分のメモに書いたことを調べて"
- "以前調べたこと覚えてる？"
- "自分の興味を教えて"
- "自分の考えをまとめて"

## Overview

`rag` is a CLI tool that searches the user's local Markdown notes using RAG (Retrieval-Augmented Generation). It reads Markdown files, splits them into chunks, stores them in a Chroma vector DB, and retrieves relevant context to answer questions via LLM.

The user typically has two types of notes:
- `notes`: Their own words, manually written
- `agent-notes`: AI-generated or structured notes

## Prerequisites

- `rag` CLI must be installed: `uv tool install .` (in the project repo) or `pip install -e .`
- Config initialized: `rag init` (stores `notes_dirs` and `agent_notes_dirs` in `~/.config/rag/config.json`)
- `OPENAI_API_KEY` must be set in the environment

## Basic Usage

Always prefer the simplest form first:

```bash
rag "<question>"
```

Examples:

```bash
rag "私のLangChainに関する学びを教えて"
rag "最近興味を持った技術は？"
rag "以前悩んでいたことを思い出して"
```

## Options

When the user asks for something specific, you may add options:

```bash
# Include AI-generated / structured notes
rag "<question>" --agent-notes-dir ~/agent-notes

# Use a specific strategy
rag "<question>" --strategy hyde
rag "<question>" --strategy multi-query

# Use pre-built index (faster for large collections)
rag "<question>" --db-dir ~/.local/share/rag/db

# Override notes directories temporarily
rag "<question>" --notes-dir ~/work-notes --notes-dir ~/diary
```

## Strategies

| strategy | when to use |
|----------|-------------|
| `simple` | Default. Good for most direct questions. |
| `hyde` | When the question is abstract or vague. Generates a hypothetical answer to search with. |
| `multi-query` | When the question might match notes using different phrasings. Generates multiple search queries. |
| `hybrid-bm25-vector` | When keyword matching helps. Cannot be used with `--db-dir`. |
| `router` | When the question might be about general knowledge rather than personal notes. Automatically picks notes or web search. |
| `hybrid-notes-web` | When the user wants both personal notes and latest web information combined. |

## Indexing (Optional)

If the user has many notes and wants faster queries, suggest pre-building an index:

```bash
rag index --notes-dir ~/notes --agent-notes-dir ~/agent-notes --db-dir ~/.local/share/rag/db
```

Then query with:

```bash
rag "<question>" --db-dir ~/.local/share/rag/db
```

## Important Notes

- Do not confuse this with general web search. `rag` only searches the user's **local** Markdown files unless `--strategy router` or `--strategy hybrid-notes-web` is used.
- The output is plain text. If the user wants sources, mention that source paths are embedded in the context but not exposed by default in the final answer formatting.
- `rag` does not modify notes. It is read-only.
- If `rag` is not installed or configured, guide the user to run `uv tool install .` and `rag init` first.
