from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.auto import tqdm

from research_foundry.models import (
    AgentArtifact,
    IdeaCandidate,
    IdeaNoveltyAudit,
    ResearchReport,
    ResearchRequest,
    SelectionDecision,
)


class RichTqdmReporter:
    """Color terminal reporter for live Research Foundry runs."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        disable_tqdm: bool = False,
    ) -> None:
        self.console = console or Console(stderr=True, highlight=True)
        self.disable_tqdm = disable_tqdm
        self._bar: Any | None = None
        self._run_started_at: float | None = None
        self._stage_started_at: dict[str, float] = {}

    def start(self, request: ResearchRequest, total_stages: int) -> None:
        self._close_bar()
        self._run_started_at = time.monotonic()
        self._stage_started_at.clear()

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Field", request.field)
        table.add_row("Objective", request.objective)
        table.add_row("Venues", ", ".join(request.target_venues))
        table.add_row("Ideas", str(request.idea_count))
        table.add_row("Risk", request.risk_tolerance)
        table.add_row("Deep research", "enabled" if request.use_deep_research else "disabled")
        if request.constraints:
            table.add_row("Constraints", "\n".join(f"- {item}" for item in request.constraints))

        self.console.print(
            Panel(
                table,
                title="[bold bright_cyan]Research Foundry[/]",
                subtitle="[bright_magenta]live multi-agent run[/]",
                border_style="bright_cyan",
                box=box.ROUNDED,
            )
        )
        self._bar = tqdm(
            total=total_stages,
            desc="Pipeline",
            unit="stage",
            colour="cyan",
            dynamic_ncols=True,
            leave=True,
            file=self.console.file,
            disable=self.disable_tqdm,
        )

    def stage_start(self, index: int, total: int, name: str, detail: str) -> None:
        self._stage_started_at[name] = time.monotonic()
        self._set_bar(name, "starting")
        self._print(
            f"[bold bright_cyan]START[/] "
            f"[cyan]{index}/{total}[/] [bold white]{name}[/] [dim]- {detail}[/]"
        )

    def stage_tick(self, index: int, total: int, name: str) -> None:
        elapsed = self._stage_elapsed(name)
        self._set_bar(name, f"running {elapsed:,.0f}s")

    def stage_complete(
        self, index: int, total: int, name: str, artifact: AgentArtifact
    ) -> None:
        elapsed = self._stage_elapsed(name)
        summary = self._artifact_summary(artifact)
        if self._bar is not None:
            self._bar.set_postfix_str(f"done {elapsed:,.0f}s")
            self._bar.update(1)
            self._bar.refresh()
        self._print(
            f"[bold green]DONE[/]  "
            f"[green]{index}/{total}[/] [bold white]{name}[/] "
            f"[dim]- {elapsed:,.1f}s, {summary}[/]"
        )

    def stage_error(self, index: int, total: int, name: str, error: BaseException) -> None:
        self._set_bar(name, "failed")
        self._close_bar()
        self.console.print(
            Panel(
                f"[bold red]{name} failed at stage {index}/{total}[/]\n\n{error}",
                title="[bold red]Pipeline error[/]",
                border_style="red",
            )
        )

    def ideas_ready(self, ideas: list[IdeaCandidate]) -> None:
        table = Table(
            title="Generated Idea Candidates",
            box=box.SIMPLE_HEAVY,
            border_style="bright_blue",
            show_lines=True,
        )
        table.add_column("#", style="bold cyan", justify="right", no_wrap=True)
        table.add_column("Title", style="bold white", ratio=2)
        table.add_column("Thesis", style="white", ratio=4)
        table.add_column("Key Risks", style="yellow", ratio=2)

        for index, idea in enumerate(ideas, start=1):
            table.add_row(
                str(index),
                idea.title,
                self._shorten(idea.thesis, 180),
                self._shorten("; ".join(idea.key_risks[:2]), 120),
            )

        self._print(table)

    def novelty_audit_ready(self, audits: list[IdeaNoveltyAudit]) -> None:
        if not audits:
            return

        table = Table(
            title="Novelty Collision Audit",
            box=box.SIMPLE_HEAVY,
            border_style="bright_yellow",
            show_lines=True,
        )
        table.add_column("Idea", style="bold white", ratio=2)
        table.add_column("Verdict", style="bold", no_wrap=True)
        table.add_column("Scores", style="cyan", no_wrap=True)
        table.add_column("Closest Collisions", style="yellow", ratio=3)
        table.add_column("Blockers", style="red", ratio=3)

        for audit in audits:
            collisions = "; ".join(
                f"{collision.prior_work} ({collision.severity})"
                for collision in audit.closest_prior_work[:3]
            ) or "-"
            blockers = "; ".join(audit.main_track_blockers[:2]) or "-"
            table.add_row(
                audit.idea_title,
                self._verdict_text(audit.main_track_verdict),
                (
                    f"N {audit.novelty_score}/10, "
                    f"P {audit.paper_worth_score}/10, "
                    f"V {audit.venue_upside_score}/10"
                ),
                self._shorten(collisions, 180),
                self._shorten(blockers, 180),
            )

        self._print(table)

    def selection_ready(
        self, selection: SelectionDecision, selected_idea: IdeaCandidate
    ) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Selected idea", f"[bold white]{selected_idea.title}[/]")
        table.add_row("Selector score", self._score_text(selection.score))
        table.add_row("Why", self._shorten(selection.rationale, 260))
        if selection.required_next_steps:
            table.add_row(
                "Next steps",
                "\n".join(f"- {item}" for item in selection.required_next_steps[:4]),
            )

        self._print(
            Panel(
                table,
                title="[bold green]Best Idea Selector decision[/]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    def saved(self, report: ResearchReport, run_dir: object | None) -> None:
        table = Table(
            title="Saved Artifacts",
            box=box.SIMPLE_HEAVY,
            border_style="bright_magenta",
            show_lines=False,
        )
        table.add_column("Artifact", style="bold magenta", no_wrap=True)
        table.add_column("Path", style="white")

        if run_dir is not None:
            run_path = Path(run_dir)
            table.add_row("Report", str(run_path / "report.md"))
            table.add_row("JSON", str(run_path / "report.json"))
            table.add_row("DOCX", str(run_path / "selected_idea_implementation_plan.docx"))
        elif report.implementation_docx_path:
            table.add_row("DOCX", report.implementation_docx_path)
        else:
            table.add_row("Status", "No output directory recorded")

        self._print(table)

    def finish(self, report: ResearchReport) -> None:
        self._close_bar()
        elapsed = self._run_elapsed()
        top_idea = report.top_idea.title if report.top_idea else "No parsed top idea"
        score = report.selection.score if report.selection else 0

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Selected idea", f"[bold white]{top_idea}[/]")
        table.add_row("Selector score", self._score_text(score))
        table.add_row("Ideas reviewed", str(len(report.ideas)))
        table.add_row("Review rows", str(len(report.reviews)))
        table.add_row("Sources", str(len(report.literature.sources)))
        table.add_row("Elapsed", f"{elapsed:,.1f}s")

        self.console.print(
            Panel(
                table,
                title="[bold green]Research Foundry complete[/]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    def _set_bar(self, stage_name: str, postfix: str) -> None:
        if self._bar is None:
            return
        self._bar.set_description_str(self._shorten(stage_name, 28))
        self._bar.set_postfix_str(postfix)
        self._bar.refresh()

    def _print(self, *objects: object) -> None:
        context = (
            tqdm.external_write_mode(file=self.console.file)
            if self._bar is not None
            else nullcontext()
        )
        with context:
            self.console.print(*objects)

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _stage_elapsed(self, name: str) -> float:
        started_at = self._stage_started_at.get(name)
        if started_at is None:
            return 0.0
        return time.monotonic() - started_at

    def _run_elapsed(self) -> float:
        if self._run_started_at is None:
            return 0.0
        return time.monotonic() - self._run_started_at

    @staticmethod
    def _artifact_summary(artifact: AgentArtifact) -> str:
        words = len(artifact.content.split())
        parts = [f"{words:,} words"]
        if artifact.sources:
            parts.append(f"{len(artifact.sources):,} sources")

        usage = artifact.metadata.get("usage")
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens")
            if isinstance(total_tokens, int):
                parts.append(f"{total_tokens:,} tokens")

        if artifact.response_id:
            parts.append(f"id {artifact.response_id[-8:]}")
        return ", ".join(parts)

    @staticmethod
    def _score_text(score: int) -> str:
        if score >= 8:
            return f"[bold green]{score}/10[/]"
        if score >= 6:
            return f"[bold yellow]{score}/10[/]"
        return f"[bold red]{score}/10[/]"

    @staticmethod
    def _verdict_text(verdict: str) -> str:
        if verdict == "pass":
            return "[bold green]pass[/]"
        if verdict == "borderline":
            return "[bold yellow]borderline[/]"
        return "[bold red]fail[/]"

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
