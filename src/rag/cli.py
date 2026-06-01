import argparse
import shutil
import sys
from pathlib import Path

from rag.config import (
    get_agent_notes_dirs,
    get_config_value,
    get_default_skills_dir,
    get_notes_dirs,
    load_config,
    save_config,
)
from rag.core import (
    build_chat_model,
    build_vectorstore,
    load_markdown_documents,
    load_vectorstore,
    split_documents,
)
from rag.strategies import build_chain, list_strategies


def _exit_with(message: str, code: int = 1):
    print(message, file=sys.stderr)
    sys.exit(code)


def _load_all_documents(args) -> list:
    note_dirs = get_notes_dirs(getattr(args, "notes_dir", None))
    agent_dirs = get_agent_notes_dirs(getattr(args, "agent_notes_dir", None))

    documents: list = []
    if note_dirs:
        documents.extend(load_markdown_documents(note_dirs, source_type="notes"))
    if agent_dirs:
        documents.extend(load_markdown_documents(agent_dirs, source_type="agent-notes"))
    return documents


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_init(_args):
    print("=== rag 初期設定 ===")

    dirs_input = input("自分の言葉で書いたノートのディレクトリ（カンマ区切り）: ").strip()
    if not dirs_input:
        _exit_with("ディレクトリが指定されていません。")
    notes_dirs = [d.strip() for d in dirs_input.split(",") if d.strip()]

    agent_dirs_input = input("AI生成・構造化ノートのディレクトリ（カンマ区切り、無ければEnter）: ").strip()
    agent_notes_dirs = [d.strip() for d in agent_dirs_input.split(",") if d.strip()] if agent_dirs_input else []

    print(f"\n利用可能な strategy: {', '.join(list_strategies())}")
    strategy_input = input("デフォルトの strategy を入力 [simple]: ").strip()
    default_strategy = strategy_input or "simple"

    config = {
        "notes_dirs": notes_dirs,
        "agent_notes_dirs": agent_notes_dirs,
        "default_strategy": default_strategy,
    }
    save_config(config)
    print("\n設定を保存しました")
    print(f"  notes_dirs: {notes_dirs}")
    if agent_notes_dirs:
        print(f"  agent_notes_dirs: {agent_notes_dirs}")
    print(f"  default_strategy: {default_strategy}")


def cmd_index(args):
    note_dirs = get_notes_dirs(args.notes_dir)
    agent_dirs = get_agent_notes_dirs(args.agent_notes_dir)
    if not note_dirs and not agent_dirs:
        _exit_with("error: --notes-dir / --agent-notes-dir が指定されていないか、configに設定されていません。\n"
                   "  ヒント: rag init で設定するか、--notes-dir / --agent-notes-dir を指定してください。")
    if not args.db_dir:
        _exit_with("error: --db-dir を指定してください。")

    documents = _load_all_documents(args)
    if not documents:
        _exit_with("error: 読み込めるドキュメントがありませんでした。")

    chunks = split_documents(documents)
    build_vectorstore(chunks, persist_directory=args.db_dir)
    print(f"Done. Indexed {len(chunks)} chunks from {len(documents)} docs.")


def cmd_query(args):
    note_dirs = get_notes_dirs(args.notes_dir)
    agent_dirs = get_agent_notes_dirs(args.agent_notes_dir)

    if not note_dirs and not agent_dirs and not args.db_dir:
        _exit_with("error: notes-dir / agent-notes-dir が指定されていないか、configに設定されていません。\n"
                   "  ヒント: rag init で設定するか、--notes-dir / --agent-notes-dir / --db-dir を指定してください。")

    strategy = args.strategy or get_config_value("default_strategy", "simple")
    if strategy not in list_strategies():
        _exit_with(f"error: 不明な strategy '{strategy}'.\n"
                   f"  利用可能: {', '.join(list_strategies())}")

    model = build_chat_model()

    if args.db_dir:
        db = load_vectorstore(args.db_dir)
        chunks = None
    else:
        documents = _load_all_documents(args)
        if not documents:
            _exit_with("error: 読み込めるドキュメントがありませんでした。")
        chunks = split_documents(documents)
        db = build_vectorstore(chunks)

    # hybrid-bm25-vector は chunks が必要
    if strategy == "hybrid-bm25-vector" and chunks is None:
        _exit_with("error: strategy 'hybrid-bm25-vector' は --notes-dir / --agent-notes-dir でのリアルタイム構築が必要です。\n"
                   "  --db-dir を使う場合はこの strategy は使えません。")

    chain = build_chain(strategy=strategy, db=db, chunks=chunks, model=model)
    result = chain.invoke(args.question)
    print(result)


def _find_repo_skills_dir() -> Path | None:
    """Try to find skills/rag-notes/ relative to this installed package."""
    # When installed via uv tool, the package is in site-packages.
    # We can't reliably find the original repo from there.
    # Check if running from the source repo directly.
    this_file = Path(__file__).resolve()
    # src/rag/cli.py -> repo root
    if this_file.parts[-3:] == ("src", "rag", "cli.py"):
        repo_root = this_file.parents[2]
        skill_src = repo_root / "skills" / "rag-notes"
        if skill_src.exists():
            return skill_src
    return None


def cmd_install_skill(args):
    # --- source ---
    if args.repo_dir:
        skill_src = Path(args.repo_dir).expanduser().resolve() / "skills" / "rag-notes"
    else:
        skill_src = _find_repo_skills_dir()

    if not skill_src or not skill_src.exists():
        _exit_with(
            "error: skills/rag-notes ディレクトリが見つかりませんでした。\n"
            "  ヒント: --repo-dir でリポジトリのパスを指定してください。\n"
            "  例: rag install-skill --repo-dir ~/dev/langchain-rag-notes"
        )

    # --- destination ---
    if args.skills_dir:
        skills_dir = Path(args.skills_dir).expanduser().resolve()
    else:
        skills_dir = get_default_skills_dir()

    if not skills_dir:
        _exit_with(
            "error: pi-agent の skills ディレクトリが自動検出できませんでした。\n"
            "  ヒント: --skills-dir でパスを指定するか、PI_SKILLS_DIR 環境変数を設定してください。"
        )

    dest = skills_dir / "rag-notes"

    if dest.exists():
        if args.force:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        else:
            _exit_with(
                f"error: {dest} は既に存在します。\n"
                f"  上書きする場合は --force を付けてください。"
            )

    skills_dir.mkdir(parents=True, exist_ok=True)

    if args.link:
        dest.symlink_to(skill_src.resolve(), target_is_directory=True)
        print(f"Linked: {dest} -> {skill_src}")
    else:
        shutil.copytree(skill_src, dest)
        print(f"Copied: {skill_src} -> {dest}")

    print("\npi-agent に rag-notes スキルが認識されるようになりました。")
    print("pi を再起動するか、スキルをリロードしてください。")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) >= 2 and sys.argv[1] in ("init", "index", "install-skill"):
        command = sys.argv.pop(1)

        if command == "init":
            parser = argparse.ArgumentParser(prog="rag init", description="対話式で初期設定を行います。")
            args = parser.parse_args()
            cmd_init(args)

        elif command == "index":
            parser = argparse.ArgumentParser(prog="rag index", description="ノートをインデックス化して永続化します。")
            parser.add_argument("--notes-dir", action="append", help="自分の言葉のMarkdownノートディレクトリ（複数可）")
            parser.add_argument("--agent-notes-dir", action="append", help="AI生成・構造化ノートのディレクトリ（複数可）")
            parser.add_argument("--db-dir", required=True, help="Chroma DB出力ディレクトリ")
            args = parser.parse_args()
            cmd_index(args)

        elif command == "install-skill":
            parser = argparse.ArgumentParser(prog="rag install-skill", description="pi-agent に rag-notes スキルをインストールします。")
            parser.add_argument("--repo-dir", help="langchain-rag-notes リポジトリのパス（デフォルト: 自動検出）")
            parser.add_argument("--skills-dir", help="pi-agent skills ディレクトリのパス（デフォルト: 自動検出）")
            parser.add_argument("--link", action="store_true", help="シンボリックリンクで配置（デフォルトはコピー）")
            parser.add_argument("--force", action="store_true", help="既存のスキルを上書き")
            args = parser.parse_args()
            cmd_install_skill(args)

    else:
        # Default: query mode. Everything positional becomes the question.
        parser = argparse.ArgumentParser(prog="rag", description="LangChain RAG CLI for local Markdown notes.")
        parser.add_argument("question", nargs="+", help="質問文")
        parser.add_argument("--notes-dir", action="append", help="自分の言葉のMarkdownノートディレクトリ（複数可）")
        parser.add_argument("--agent-notes-dir", action="append", help="AI生成・構造化ノートのディレクトリ（複数可）")
        parser.add_argument("--db-dir", help="事前構築したChroma DBディレクトリ")
        parser.add_argument("--strategy", choices=list_strategies(), help="RAG戦略")
        args = parser.parse_args()
        args.question = " ".join(args.question)
        cmd_query(args)


if __name__ == "__main__":
    main()
