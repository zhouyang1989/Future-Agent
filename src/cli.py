#!/usr/bin/env python3
"""
cli.py
Future-Agent 命令行入口（Rich 美化版）。
v1.5 新增：Self-RAG 批判式推荐路径（--self-rag）。
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import Config, load_config, resolve_config_path



# ------------------------------------------------------------------------------
# 日志与控制台
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# 抑制 httpx/openai 的 HTTP 请求 INFO 日志，避免刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logger = logging.getLogger("future-agent")

console = Console()

# ------------------------------------------------------------------------------
# Typer App
# ------------------------------------------------------------------------------
app = typer.Typer(
    name="future-agent",
    help="🔮 Future-Agent —— 连接你的知识库与外部世界",
    no_args_is_help=True,
    add_completion=False,
)


def _print_banner() -> None:
    """打印启动横幅。"""
    banner = Text()
    banner.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="bold cyan")
    banner.append("┃  ", style="bold cyan")
    banner.append("🔮  Future-Agent", style="bold bright_cyan")
    banner.append("  v1.5          ┃\n", style="bold cyan")
    banner.append("┃  连接你的知识库与外部世界            ┃\n", style="dim cyan")
    banner.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛", style="bold cyan")
    console.print(banner)
    console.print()


def _ensure_runtime_dirs() -> None:
    """确保运行时数据目录存在。"""
    dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "chroma",
        PROJECT_ROOT / "data" / "reports",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _load_config_safe() -> Config:
    try:
        return load_config()
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ {e}[/bold red]")
        raise typer.Exit(1)


def _resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    return Path(path_str).expanduser().resolve()


# ------------------------------------------------------------------------------
# 公共辅助函数
# ------------------------------------------------------------------------------
def _prepare_output_dir(output: Optional[str]) -> Path:
    """解析并确保报告输出目录存在。"""
    output_dir = _resolve_path(output) or (PROJECT_ROOT / "data" / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_inbox_path(config: Config) -> Optional[Path]:
    """解析 Obsidian Inbox 路径。"""
    return resolve_config_path(config.get("knowledge_source.obsidian.inbox_path"))


def _copy_to_inbox(report_path: Path, inbox_path: Optional[Path]) -> None:
    """将报告复制到 Obsidian Inbox。"""
    if not inbox_path:
        return
    try:
        import shutil

        inbox_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, inbox_path / report_path.name)
        console.print(f"[dim]📋 已复制到 Inbox: {inbox_path / report_path.name}[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠️ 复制到 Inbox 失败: {e}[/yellow]")


# ------------------------------------------------------------------------------
# v1.5 Self-RAG 展示层辅助函数
# ------------------------------------------------------------------------------
def _render_self_rag_result(
    recommendations: List[Any],
    report_path: Path,
    mode: str,
    beam_size: int,
    date_str: str,
) -> None:
    """渲染 Self-RAG 推荐结果表格与结果面板。"""
    console.print()
    table = Table(
        title=f"📚 Self-RAG 批判式推荐日报 ({date_str}) —— {mode} 模式",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("类型", width=10)
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("标题", min_width=30)
    table.add_column("相似度", justify="right", width=8)
    table.add_column("支持度", width=12)
    table.add_column("效用", justify="center", width=6)
    table.add_column("关联笔记", style="dim", width=16)

    bucket_style = {
        "strong": "[bold green]⭐ 强相关[/bold green]",
        "boundary": "[bold yellow]🌱 边界[/bold yellow]",
        "cross": "[bold blue]🚀 跨界[/bold blue]",
    }

    support_style = {
        "Fully Supported": "[bold green]✅ 完全支持[/bold green]",
        "Partially Supported": "[bold yellow]⚠️ 部分支持[/bold yellow]",
        "No Support": "[bold red]❌ 无支持[/bold red]",
        "Contradictory": "[bold red]⛔ 矛盾[/bold red]",
        "Skipped": "[dim]⏭️ 跳过[/dim]",
        "-": "[dim]-[/dim]",
    }

    for i, rec in enumerate(recommendations, 1):
        meta = rec.metadata or {}
        support = meta.get("support_level", "-")
        utility = meta.get("utility_score", "-")
        table.add_row(
            str(i),
            bucket_style.get(rec.diversity_bucket, rec.diversity_bucket),
            rec.arxiv_id,
            rec.title[:45] + "..." if len(rec.title) > 45 else rec.title,
            f"{rec.similarity:.3f}",
            support_style.get(support, support),
            str(utility),
            rec.matched_note[:14] + "..." if len(rec.matched_note) > 14 else rec.matched_note,
        )

    console.print(table)
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✅ Self-RAG 报告已生成[/bold green]\n\n"
            f"  [dim]路径:[/dim] [cyan]{report_path}[/cyan]\n"
            f"  [dim]模式:[/dim] [cyan]{mode}[/cyan]\n"
            f"  [dim]数量:[/dim] [cyan]{len(recommendations)}[/cyan] 篇",
            border_style="green",
            padding=(1, 2),
        )
    )


def _render_v1_result(
    recommendations: List[Any],
    report_path: Path,
    diversity: str,
    date_str: str,
) -> None:
    """渲染 v1.0 推荐结果表格与结果面板。"""
    console.print()
    table = Table(
        title=f"📚 今日推荐日报 ({date_str})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("类型", width=10)
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("标题", min_width=30)
    table.add_column("相似度", justify="right", width=8)
    table.add_column("关联笔记", style="dim", width=16)

    bucket_style = {
        "strong": "[bold green]⭐ 强相关[/bold green]",
        "boundary": "[bold yellow]🌱 边界[/bold yellow]",
        "cross": "[bold blue]🚀 跨界[/bold blue]",
    }

    for i, rec in enumerate(recommendations, 1):
        table.add_row(
            str(i),
            bucket_style.get(rec.diversity_bucket, rec.diversity_bucket),
            rec.arxiv_id,
            rec.title[:45] + "..." if len(rec.title) > 45 else rec.title,
            f"{rec.similarity:.3f}",
            rec.matched_note[:14] + "..." if len(rec.matched_note) > 14 else rec.matched_note,
        )

    console.print(table)
    console.print()

    # 文件输出面板
    console.print(
        Panel.fit(
            f"[bold green]✅ 报告已生成[/bold green]\n\n"
            f"  [dim]路径:[/dim] [cyan]{report_path}[/cyan]\n"
            f"  [dim]策略:[/dim] [cyan]{diversity}[/cyan]\n"
            f"  [dim]数量:[/dim] [cyan]{len(recommendations)}[/cyan] 篇",
            border_style="green",
            padding=(1, 2),
        )
    )


# ------------------------------------------------------------------------------
# ingest
# ------------------------------------------------------------------------------
@app.command()
def ingest(
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="知识库路径，覆盖配置文件"
    ),
    reader: Optional[str] = typer.Option(
        None, "--reader", "-r", help="读取器类型: obsidian / markdown"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="强制全量重建索引"
    ),
):
    """将本地笔记库解析、分块、嵌入并索引到向量存储。"""
    _print_banner()
    _ensure_runtime_dirs()

    config = _load_config_safe()

    source_path = _resolve_path(source)
    if not source_path:
        source_path = resolve_config_path(config.get("knowledge_source.obsidian.vault_path"))
    if not source_path:
        source_path = resolve_config_path(config.get("knowledge_source.markdown_folder.folder_path"))

    reader_type = reader or config.get("knowledge_source.reader_type", "obsidian")

    if not source_path or not source_path.exists():
        console.print(
            "[bold red]❌ 未找到有效的知识库路径。[/bold red]\n"
            "   请使用 [cyan]--source[/cyan] 指定，或在 [cyan]config.local.yaml[/cyan] 中配置路径。"
        )
        raise typer.Exit(1)

    # 阶段进度
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Stage 1: 读取
        task_read = progress.add_task("[cyan]📖 读取笔记库...", total=None)
        from src.readers import get_reader

        if reader_type == "obsidian":
            reader_config = config.get("knowledge_source.obsidian")
        elif reader_type == "markdown":
            reader_config = config.get("knowledge_source.markdown_folder")
        else:
            reader_config = None

        reader_inst = get_reader(reader_type, reader_config)
        documents = list(reader_inst.read(str(source_path)))
        progress.update(task_read, completed=1, total=1)

        if not documents:
            console.print("[yellow]⚠️ 未找到任何文档，请检查路径或 ignore_folders 配置[/yellow]")
            raise typer.Exit(0)

        # Stage 2: 分块
        from src.processors.chunker import Chunker

        task_chunk = progress.add_task("[cyan]✂️ 文本分块...", total=len(documents))
        chunker = Chunker(config.raw)
        chunks = chunker.chunk(documents)
        progress.update(task_chunk, completed=len(documents), total=len(documents))

        # Stage 3: 嵌入（关闭内部 progress bar，由 Rich 接管）
        from src.embeddings.embedder import Embedder

        task_embed = progress.add_task("[cyan]🔢 SBERT 嵌入...", total=len(chunks))
        embedder = Embedder(config.raw)
        embeddings = embedder.encode(chunks, show_progress=False)
        progress.update(task_embed, completed=len(chunks), total=len(chunks))

        # Stage 4: 存储
        from src.storage.vector_store import VectorStore

        task_store = progress.add_task("[cyan]💾 存入向量库...", total=1)
        store = VectorStore(config.raw)
        store.upsert_notes(chunks, embeddings, force=force)
        progress.update(task_store, completed=1, total=1)

    # 结果面板
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✅ 知识库摄入完成[/bold green]\n\n"
            f"  [dim]读取器 :[/dim] [cyan]{reader_type}[/cyan]\n"
            f"  [dim]文档数 :[/dim] [cyan]{len(documents)}[/cyan] 篇\n"
            f"  [dim]文本块 :[/dim] [cyan]{len(chunks)}[/cyan] 个\n"
            f"  [dim]向量维度:[/dim] [cyan]{embeddings.shape[1]}[/cyan] 维\n"
            f"  [dim]目标集合:[/dim] [cyan]{store.notes_collection_name}[/cyan]\n"
            f"  [dim]索引模式:[/dim] [cyan]{'全量重建' if force else '增量更新'}[/cyan]",
            title="ingest 结果",
            border_style="green",
            padding=(1, 2),
        )
    )


# ------------------------------------------------------------------------------
# fetch
# ------------------------------------------------------------------------------
@app.command()
def fetch(
    days: int = typer.Option(7, "--days", "-d", min=1, max=30),
    max_results: int = typer.Option(100, "--max-results", "-n", min=1, max=500),
    no_index: bool = typer.Option(False, "--no-index", help="仅缓存，不向量化"),
):
    """从 arXiv 按分类批量获取最新论文，并索引到向量库。"""
    _print_banner()
    _ensure_runtime_dirs()

    config = _load_config_safe()
    arxiv_config = config.get("arxiv") or {}
    categories = arxiv_config.get("categories", ["cs.CL", "cs.LG", "cs.IR"])

    console.print(
        f"[dim]📚 分类:[/dim] [cyan]{', '.join(categories)}[/cyan]  "
        f"[dim]📅 最近[/dim] [cyan]{days}[/cyan] 天  "
        f"[dim]🔢 上限[/dim] [cyan]{max_results}[/cyan] 篇"
    )
    console.print()

    from src.fetchers.arxiv_fetcher import ArxivFetcher

    # arXiv 请求阶段用 spinner（总数量未知）
    with console.status("[bold green]🌐 正在请求 arXiv API...[/bold green]", spinner="dots"):
        fetcher = ArxivFetcher(arxiv_config)
        papers = fetcher.fetch(days=days, max_results=max_results)

    if not papers:
        console.print("[yellow]⚠️ 未获取到任何论文，请检查分类配置或网络连接[/yellow]")
        raise typer.Exit(0)

    if no_index:
        console.print(
            Panel.fit(
                f"[bold green]✅ 论文已缓存[/bold green]\n\n"
                f"  [dim]数量:[/dim] [cyan]{len(papers)}[/cyan] 篇\n"
                f"  [dim]模式:[/dim] [cyan]--no-index（跳过向量化）[/cyan]",
                border_style="green",
            )
        )
        raise typer.Exit(0)

    # 向量化阶段用 Progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_convert = progress.add_task("[cyan]📄 转换为文本块...", total=1)
        chunks = fetcher.to_chunks(papers)
        progress.update(task_convert, completed=1, total=1)

        from src.embeddings.embedder import Embedder

        task_embed = progress.add_task("[cyan]🔢 SBERT 嵌入...", total=len(chunks))
        embedder = Embedder(config.raw)
        embeddings = embedder.encode(chunks, show_progress=False)
        progress.update(task_embed, completed=len(chunks), total=len(chunks))

        from src.storage.vector_store import VectorStore

        task_store = progress.add_task("[cyan]💾 存入向量库...", total=1)
        store = VectorStore(config.raw)
        store.upsert_papers(chunks, embeddings)
        progress.update(task_store, completed=1, total=1)

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✅ 论文抓取与索引完成[/bold green]\n\n"
            f"  [dim]获取数量:[/dim] [cyan]{len(papers)}[/cyan] 篇\n"
            f"  [dim]文本块数:[/dim] [cyan]{len(chunks)}[/cyan] 个\n"
            f"  [dim]向量维度:[/dim] [cyan]{embeddings.shape[1]}[/cyan] 维\n"
            f"  [dim]目标集合:[/dim] [cyan]{store.papers_collection_name}[/cyan]",
            title="fetch 结果",
            border_style="green",
            padding=(1, 2),
        )
    )


# ------------------------------------------------------------------------------
# recommend（v1.0 + v1.5 Self-RAG 双路径）
# ------------------------------------------------------------------------------
@app.command()
def recommend(
        output: Optional[str] = typer.Option(
            None, "--output", "-o", help="报告输出目录"
        ),
        to_obsidian: bool = typer.Option(
            False, "--to-obsidian", help="同时写入 Obsidian Inbox"
        ),
        top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=20),
        diversity: str = typer.Option("boundary_mix", "--diversity"),
        # v1.5 Self-RAG 参数
        self_rag: bool = typer.Option(
            False, "--self-rag", help="启用 Self-RAG 批判式推荐（v1.5）"
        ),
        mode: str = typer.Option(
            "balanced", "--mode",
            help="Self-RAG 模式: strict(严格事实) / balanced(平衡) / creative(创意) / fast(省流)"
        ),
        beam_size: int = typer.Option(
            2, "--beam-size", "-b", min=1, max=5,
            help="Segment-level Beam Search 宽度（1=Greedy）"
        ),
):
    """
    基于知识库语义状态，生成今日 arXiv 论文推荐日报。

    v1.0（默认）: 固定检索 + 相似度匹配 + 多样性约束
    v1.5（--self-rag）: 固定检索 + 批判式筛选 + 自我校验 + 可控解码
    """
    _print_banner()
    _ensure_runtime_dirs()

    config = _load_config_safe()

    output_dir = _prepare_output_dir(output)
    inbox_path = _resolve_inbox_path(config) if to_obsidian else None
    if to_obsidian and inbox_path:
        console.print(f"[dim]📝 Inbox:[/dim] [cyan]{inbox_path}[/cyan]")
    elif to_obsidian:
        console.print("[yellow]⚠️ 未配置 inbox_path，忽略 --to-obsidian[/yellow]")

    date_str = datetime.now().strftime("%Y-%m-%d")

    if self_rag:
        console.print("[bold cyan]🧠 启用 Self-RAG 批判式推荐（v1.5）[/bold cyan]")
        console.print(f"[dim]   模式: {mode} | Beam: {beam_size} | Top-K: {top_k}[/dim]\n")

        from src.engine.self_rag.self_rag_recommender import (
            NoRecommendationError,
            SelfRAGRecommendEngine,
        )

        engine = SelfRAGRecommendEngine(config.raw, mode, beam_size, top_k)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_isrel = progress.add_task("[cyan]🔍 IsRel 批判...", total=None)
            task_generate = progress.add_task("[cyan]✍️ 生成推荐理由...", total=None, visible=False)

            def _cb(step: str, current: int, total: int):
                if step == "isrel":
                    progress.update(task_isrel, total=total, completed=current)
                elif step == "generate":
                    if current == 1:
                        progress.update(task_isrel, visible=False)
                        progress.update(task_generate, visible=True)
                    progress.update(task_generate, total=total, completed=current)

            try:
                recommendations, report_path = engine.run(
                    output_dir, date_str, progress_callback=_cb
                )
            except NoRecommendationError:
                console.print(
                    "[yellow]⚠️ Self-RAG 未生成任何有效推荐，"
                    "请检查笔记库或 arXiv 索引状态[/yellow]"
                )
                raise typer.Exit(0)

        _copy_to_inbox(report_path, inbox_path)
        _render_self_rag_result(recommendations, report_path, mode, beam_size, date_str)
        return

    # v1.0 传统路径
    from src.engine.legacy_recommender import LegacyRecommendEngine

    valid = {"strong_only", "boundary_mix", "cross_domain"}
    if diversity not in valid:
        console.print(f"[bold red]❌ 无效策略: {diversity}。可选: {valid}[/bold red]")
        raise typer.Exit(1)

    engine = LegacyRecommendEngine(config.raw)
    recommendations, report_path = engine.run(
        top_k=top_k, diversity=diversity, output_dir=output_dir, date_str=date_str
    )

    if not recommendations:
        console.print("[yellow]⚠️ 未生成任何推荐，请确认已执行 ingest 和 fetch[/yellow]")

    _copy_to_inbox(report_path, inbox_path)
    _render_v1_result(recommendations, report_path, diversity, date_str)


# ------------------------------------------------------------------------------
# status
# ------------------------------------------------------------------------------
@app.command()
def status():
    """查看系统状态与向量库统计。"""
    _print_banner()

    config = _load_config_safe()

    # 配置表格
    cfg_table = Table(title="⚙️ 配置概览", box=box.SIMPLE_HEAD)
    cfg_table.add_column("项", style="dim")
    cfg_table.add_column("值", style="cyan")
    cfg_table.add_row("知识源类型", config.get("knowledge_source.reader_type", "-"))
    cfg_table.add_row("arXiv 分类", ", ".join(config.get("arxiv.categories", [])))
    cfg_table.add_row("嵌入模型", config.get("embedding.model_name", "-"))
    cfg_table.add_row("计算设备", config.get("embedding.device", "auto"))
    cfg_table.add_row("距离函数", config.get("vector_store.distance_fn", "cosine"))

    # v1.5 配置展示
    if config.get("self_rag.enabled"):
        cfg_table.add_row("Self-RAG", "[green]启用[/green]")
        cfg_table.add_row("LLM Provider", config.get("self_rag.llm.provider", "-"))
        cfg_table.add_row("LLM Model", config.get("self_rag.llm.model", "-"))
    else:
        cfg_table.add_row("Self-RAG", "[dim]未启用[/dim]")

    console.print(cfg_table)
    console.print()

    # 目录与统计表格
    from src.storage.vector_store import VectorStore

    store = VectorStore(config.raw)
    stats = store.stats()

    stat_table = Table(title="📊 向量库统计", box=box.SIMPLE_HEAD)
    stat_table.add_column("集合", style="dim")
    stat_table.add_column("状态", justify="right")
    stat_table.add_row("笔记 (my_notes)", f"[green]{stats['notes_count']}[/green] 条向量")
    stat_table.add_row("论文 (arxiv_papers)", f"[green]{stats['papers_count']}[/green] 条向量")

    console.print(stat_table)
    console.print()

    # 目录检查
    dir_table = Table(title="📁 运行时目录", box=box.SIMPLE_HEAD)
    dir_table.add_column("路径")
    dir_table.add_column("状态", justify="center")
    for subpath, label in [
        ("data/raw", "原始数据"),
        ("data/processed", "处理后数据"),
        ("data/chroma", "向量库"),
        ("data/reports", "报告"),
    ]:
        p = PROJECT_ROOT / subpath
        emoji = "✅" if p.exists() else "❌"
        dir_table.add_row(subpath, emoji)

    console.print(dir_table)


# ------------------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app()