import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich import print as rprint

from .main import annotate

console = Console()
CONFIG_FILE = Path.home() / ".glosser_config"



def save_api_key(groq_api_key: str) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump({"GROQ_API_KEY": groq_api_key}, f)


def load_api_key() -> str | None:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            return config.get("GROQ_API_KEY")
    return None


STEPS = [
    "Scaling PDF pages",
    "Building references database",
    "Annotating references",
    "Scanning for abbreviations",
    "Looking up full forms",
    "Annotating abbreviations",
    "Saving annotated PDF",
]


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


async def async_main() -> None:
    console.print(
        Panel.fit(
            "[bold white]glosser[/bold white]  [dim]: research paper annotator[/dim]\n"
            "[dim]Adds citation titles & abbreviation expansions to your PDF margins.[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()

    groq_api_key = load_api_key()
    if groq_api_key:
        console.print("[green]✓[/green] Using saved GROQ API key.")
    else:
        groq_api_key = Prompt.ask("[bold]Enter your GROQ API key[/bold]", password=True, console=console)
        save_api_key(groq_api_key)
        console.print("[green]✓[/green] API key saved for future use.")

    console.print()

    pdf_path = Prompt.ask(
        r"[bold]PDF path[/bold] (D:\something\paper.pdf)",
        console=console,
    )
    console.print()

    with make_progress() as progress:
        task_ids: dict[str, int] = {}
        for step in STEPS:
            tid = progress.add_task(step, total=100, visible=False)
            task_ids[step] = tid

        current_step: list[str] = [None]

        def on_progress(step: str, done: int, total: int) -> None:
            tid = task_ids.get(step)
            if tid is None:
                return

            if current_step[0] and current_step[0] != step:
                prev_tid = task_ids[current_step[0]]
                progress.update(prev_tid, completed=100, visible=True)

            current_step[0] = step
            pct = int((done / total) * 100) if total else 100
            progress.update(tid, completed=pct, visible=True)

        try:
            result = await annotate(
                path=pdf_path,
                GROQ_API_KEY=groq_api_key,
                progress_callback=on_progress,
            )
            if current_step[0]:
                progress.update(task_ids[current_step[0]], completed=100, visible=True)

        except Exception as exc:
            console.print()
            console.print(f"[bold red]✗ Error:[/bold red] {exc}")
            return

    out_path, annotations_added = result
    console.print()
    console.print(
        Panel(
            f"[bold green]✓ Done![/bold green]  Added [bold]{annotations_added}[/bold] annotation(s).\n"
            f"[dim]Saved to:[/dim] [cyan]{out_path}[/cyan]",
            border_style="green",
            padding=(0, 2),
        )
    )


def main() -> None:
    asyncio.run(async_main())
