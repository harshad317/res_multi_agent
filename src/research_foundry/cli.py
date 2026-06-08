from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.panel import Panel

from research_foundry.config import Settings
from research_foundry.gateway import DryRunGateway, OpenAIResponsesGateway
from research_foundry.models import ResearchReport, ResearchRequest
from research_foundry.pipeline import ResearchFoundry
from research_foundry.storage import RunStore
from research_foundry.terminal import RichTqdmReporter


DEFAULT_NOVELTY_PASS_BATCHES = 3
DEFAULT_SELECTOR_SCORE_BATCHES = 3
DEFAULT_SELECTOR_SCORE_THRESHOLD = 8


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def score_1_to_10(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 10:
        raise argparse.ArgumentTypeError("must be between 1 and 10")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-foundry",
        description="Run a multi-agent research idea foundry.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the research foundry pipeline.")
    run.add_argument("--field", required=True, help="Research field or subfield.")
    run.add_argument("--objective", required=True, help="What the system should optimize for.")
    run.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Constraint to include. Repeat this flag for multiple constraints.",
    )
    run.add_argument(
        "--venue",
        action="append",
        default=[],
        help="Target venue. Defaults to ICLR and NeurIPS.",
    )
    run.add_argument("--ideas", type=int, default=3, help="Number of ideas to generate.")
    run.add_argument(
        "--risk-tolerance",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
    )
    run.add_argument(
        "--ambition-floor",
        type=int,
        default=8,
        choices=range(1, 11),
        metavar="{1..10}",
        help="Minimum expected upside score for generated and selected ideas.",
    )
    run.add_argument("--fast", action="store_true", help="Use the faster deep research model.")
    run.add_argument("--dry-run", action="store_true", help="Run without OpenAI API calls.")
    run.add_argument(
        "--no-deep-research",
        action="store_true",
        help="Use GPT-5.5 web search instead of the specialized deep research model.",
    )
    run.add_argument("--out-dir", default=None, help="Directory for run artifacts.")
    run.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the Rich/tqdm live terminal display.",
    )
    run.add_argument(
        "--max-wait-seconds",
        type=positive_int,
        default=None,
        help=(
            "Maximum seconds to wait for a background Responses API stage. "
            "Defaults to RESEARCH_FOUNDRY_MAX_WAIT_SECONDS."
        ),
    )
    run.add_argument(
        "--until-novelty-pass",
        nargs="?",
        const=DEFAULT_NOVELTY_PASS_BATCHES,
        default=1,
        type=positive_int,
        metavar="MAX_BATCHES",
        help=(
            "Keep running fresh idea batches until at least one candidate passes the "
            f"novelty-collision audit. With no value, tries {DEFAULT_NOVELTY_PASS_BATCHES} "
            "batches; with a value, tries up to MAX_BATCHES."
        ),
    )
    run.add_argument(
        "--until-selector-score",
        nargs="?",
        const=DEFAULT_SELECTOR_SCORE_THRESHOLD,
        default=None,
        type=score_1_to_10,
        metavar="MIN_SCORE",
        help=(
            "Keep running fresh idea batches until the Best Idea Selector score is at "
            "least MIN_SCORE. The effective threshold is never below --ambition-floor. "
            f"With no value, uses max({DEFAULT_SELECTOR_SCORE_THRESHOLD}/10, "
            f"--ambition-floor) and tries up to {DEFAULT_SELECTOR_SCORE_BATCHES} batches."
        ),
    )
    return parser


def build_request(
    args: argparse.Namespace, *, retry_constraints: list[str] | None = None
) -> ResearchRequest:
    return ResearchRequest(
        field=args.field,
        objective=args.objective,
        constraints=[*args.constraint, *(retry_constraints or [])],
        target_venues=args.venue or ["ICLR", "NeurIPS"],
        idea_count=args.ideas,
        risk_tolerance=args.risk_tolerance,
        ambition_floor=args.ambition_floor,
        use_deep_research=not args.no_deep_research,
        fast=args.fast,
        dry_run=args.dry_run,
    )


def report_has_novelty_pass(report: ResearchReport) -> bool:
    floor = report.request.ambition_floor
    return any(
        audit.main_track_verdict == "pass"
        and audit.novelty_score >= floor
        and audit.paper_worth_score >= floor
        and audit.venue_upside_score >= floor
        for audit in report.novelty_audits
    )


def report_clears_retry_gates(
    report: ResearchReport,
    *,
    require_novelty_pass: bool,
    selector_score_threshold: int | None,
) -> bool:
    novelty_cleared = not require_novelty_pass or report_has_novelty_pass(report)
    selector_gate_cleared = ResearchFoundry.selection_gate_cleared(
        report.request, report.selection
    )
    selector_cleared = (
        selector_gate_cleared
        and (
            selector_score_threshold is None
            or report.selection.score >= selector_score_threshold
        )
    )
    return novelty_cleared and selector_cleared


def retry_target_text(
    *, require_novelty_pass: bool, selector_score_threshold: int | None
) -> str:
    targets: list[str] = []
    if require_novelty_pass:
        targets.append("at least one novelty-audit pass")
    if selector_score_threshold is not None:
        targets.append(f"selector score >= {selector_score_threshold}/10")
    return " and ".join(targets) or "one complete research batch"


def retry_failure_reasons(
    report: ResearchReport,
    *,
    require_novelty_pass: bool,
    selector_score_threshold: int | None,
) -> list[str]:
    reasons: list[str] = []
    if require_novelty_pass and not report_has_novelty_pass(report):
        reasons.append("no candidate passed the novelty-collision audit")
    if not ResearchFoundry.selection_gate_cleared(report.request, report.selection):
        reasons.append("selected idea did not clear the ambition floor")
    if (
        selector_score_threshold is not None
        and report.selection.score < selector_score_threshold
    ):
        reasons.append(
            f"selector score {report.selection.score}/10 is below "
            f"{selector_score_threshold}/10"
        )
    return reasons


def _compact_text(value: str, *, limit: int = 220) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return f"{trimmed}..."


def _unique_compact(items: list[str], *, limit: int = 4, item_chars: int = 180) -> list[str]:
    seen: set[str] = set()
    compacted: list[str] = []
    for item in items:
        compact = _compact_text(item, limit=item_chars)
        key = compact.lower()
        if not compact or key in seen:
            continue
        seen.add(key)
        compacted.append(compact)
        if len(compacted) >= limit:
            break
    return compacted


def retry_feedback_summary(report: ResearchReport) -> str:
    """Summarize failed-batch reviewer feedback for the next idea generator."""

    floor = report.request.ambition_floor
    selection = report.selection
    lines = [
        (
            f"Failed batch feedback for selected idea \"{selection.selected_title}\": "
            f"selector score {selection.score}/10; required >= {floor}/10."
        )
    ]

    dimension_failures = [
        f"{label}={value if value is not None else 'missing'}"
        for label, value in [
            ("research worth", selection.research_worth_score),
            ("paper worth", selection.paper_worth_score),
            ("venue upside", selection.venue_upside_score),
            ("fixed-pool only", selection.fixed_pool_only_score),
        ]
        if value is None or value < floor
    ]
    checklist_failures = [
        f"{check.name}={check.score}/10"
        for check in selection.main_conference_checklist
        if check.score < floor
    ]
    if dimension_failures or checklist_failures:
        lines.append(
            "- Score gaps to repair: "
            + " | ".join(_unique_compact([*dimension_failures, *checklist_failures], limit=8))
        )

    selector_risks = _unique_compact(selection.decisive_risks, limit=5)
    if selector_risks:
        lines.append("- Selector risks to fix: " + " | ".join(selector_risks))

    next_steps = _unique_compact(selection.required_next_steps, limit=5)
    if next_steps:
        lines.append("- Required repair moves: " + " | ".join(next_steps))

    selected_audit = next(
        (
            audit
            for audit in report.novelty_audits
            if audit.idea_title == selection.selected_title
        ),
        None,
    )
    if selected_audit is not None:
        lines.append(
            "- Novelty audit on selected idea: "
            f"{selected_audit.main_track_verdict}; "
            f"N/P/V={selected_audit.novelty_score}/"
            f"{selected_audit.paper_worth_score}/"
            f"{selected_audit.venue_upside_score}."
        )
        blockers = _unique_compact(selected_audit.main_track_blockers, limit=4)
        if blockers:
            lines.append("- Novelty blockers: " + " | ".join(blockers))
        collisions = _unique_compact(
            [
                f"{collision.prior_work} ({collision.severity})"
                for collision in selected_audit.closest_prior_work
            ],
            limit=4,
        )
        if collisions:
            lines.append("- Closest collisions to avoid: " + " | ".join(collisions))
        differentiators = _unique_compact(
            [
                collision.required_differentiator
                for collision in selected_audit.closest_prior_work
                if collision.required_differentiator
            ],
            limit=4,
        )
        if differentiators:
            lines.append("- Differentiators reviewers demanded: " + " | ".join(differentiators))

    selected_reviews = [
        review for review in report.reviews if review.idea_title == selection.selected_title
    ]
    fatal_flaws = _unique_compact(
        [flaw for review in selected_reviews for flaw in review.fatal_flaws],
        limit=5,
    )
    if fatal_flaws:
        lines.append("- Review fatal flaws: " + " | ".join(fatal_flaws))

    rescue_moves = _unique_compact(
        [move for review in selected_reviews for move in review.rescue_moves],
        limit=5,
    )
    if rescue_moves:
        lines.append("- Reviewer rescue moves: " + " | ".join(rescue_moves))

    batch_blockers = _unique_compact(
        [
            blocker
            for audit in report.novelty_audits
            for blocker in audit.main_track_blockers
            if audit.idea_title != selection.selected_title
        ],
        limit=5,
    )
    if batch_blockers:
        lines.append("- Repeated batch-level blockers: " + " | ".join(batch_blockers))

    lines.append(
        "- Generator mandate: do not merely avoid the old titles. The next batch must "
        "propose different mechanisms that directly repair the failure modes above; "
        "discard candidates that would likely top out at 7/10 for thin novelty, weak "
        "fixed-pool MVP, baseline vulnerability, or unclear technical differentiator."
    )
    return "\n".join(lines)


def _selection_score_gaps(report: ResearchReport) -> list[str]:
    floor = report.request.ambition_floor
    selection = report.selection
    dimension_failures = [
        f"{label}={value if value is not None else 'missing'}"
        for label, value in [
            ("research worth", selection.research_worth_score),
            ("paper worth", selection.paper_worth_score),
            ("venue upside", selection.venue_upside_score),
            ("fixed-pool only", selection.fixed_pool_only_score),
        ]
        if value is None or value < floor
    ]
    checklist_failures = [
        f"{check.name}={check.score}/10"
        for check in selection.main_conference_checklist
        if check.score < floor
    ]
    return _unique_compact([*dimension_failures, *checklist_failures], limit=8)


def _rejected_method_status(
    report: ResearchReport,
    *,
    idea_title: str,
    audit_verdict: str | None,
    selector_score_threshold: int | None,
) -> str:
    if audit_verdict == "fail":
        return "hard exclusion"
    if idea_title == report.selection.selected_title:
        selector_gate_cleared = ResearchFoundry.selection_gate_cleared(
            report.request, report.selection
        )
        if not selector_gate_cleared:
            return "hard exclusion"
    if audit_verdict == "borderline":
        return "soft warning"
    if (
        idea_title == report.selection.selected_title
        and selector_score_threshold is not None
        and report.selection.score < selector_score_threshold
    ):
        return "soft warning"
    return "reusable fragment only"


def _rejected_method_ledger(
    report: ResearchReport,
    *,
    batch_number: int,
    require_novelty_pass: bool,
    selector_score_threshold: int | None,
) -> str:
    """Build compact per-method retry memory for the next idea generator."""

    floor = report.request.ambition_floor
    audits_by_title = {audit.idea_title: audit for audit in report.novelty_audits}
    reviews_by_title = {
        idea.title: [
            review for review in report.reviews if review.idea_title == idea.title
        ]
        for idea in report.ideas
    }
    lines = [
        f"Batch {batch_number} rejected-method ledger (ambition floor {floor}/10):"
    ]

    if not report.ideas:
        lines.append("- No parseable ideas were returned; generate a fully fresh batch.")
        return "\n".join(lines)

    for index, idea in enumerate(report.ideas, start=1):
        audit = audits_by_title.get(idea.title)
        idea_reviews = reviews_by_title.get(idea.title, [])
        is_selected = idea.title == report.selection.selected_title
        status = _rejected_method_status(
            report,
            idea_title=idea.title,
            audit_verdict=audit.main_track_verdict if audit else None,
            selector_score_threshold=selector_score_threshold,
        )
        selected_text = "selected" if is_selected else "not selected"
        lines.append(f"{index}. {idea.title} ({status}; {selected_text})")
        lines.append(f"   Method: {_compact_text(idea.core_mechanism, limit=180)}")
        if idea.decisive_differentiator:
            lines.append(
                "   Claimed differentiator: "
                + _compact_text(idea.decisive_differentiator, limit=180)
            )

        reasons: list[str] = []
        avoid: list[str] = []
        reusable: list[str] = []

        if audit is None:
            reasons.append("no novelty audit was parsed for this method")
        else:
            lines.append(
                "   Gate result: "
                f"novelty {audit.main_track_verdict}, "
                f"N/P/V={audit.novelty_score}/{audit.paper_worth_score}/"
                f"{audit.venue_upside_score}"
            )
            if audit.main_track_verdict != "pass":
                reasons.append(f"novelty audit verdict was {audit.main_track_verdict}")
            audit_score_gaps = [
                label
                for label, value in [
                    ("novelty", audit.novelty_score),
                    ("paper worth", audit.paper_worth_score),
                    ("venue upside", audit.venue_upside_score),
                ]
                if value < floor
            ]
            if audit_score_gaps:
                reasons.append(
                    "audit scores below floor: " + ", ".join(audit_score_gaps)
                )
            if audit.main_track_blockers:
                reasons.extend(_unique_compact(audit.main_track_blockers, limit=3))
            collision_notes = _unique_compact(
                [
                    f"{collision.prior_work} ({collision.severity})"
                    for collision in audit.closest_prior_work
                ],
                limit=3,
            )
            if collision_notes:
                collision_text = " | ".join(collision_notes)
                reasons.append("closest collisions: " + collision_text)
                avoid.append("do not repeat the collision profile: " + collision_text)
            differentiators = _unique_compact(
                [
                    collision.required_differentiator
                    for collision in audit.closest_prior_work
                    if collision.required_differentiator
                ],
                limit=3,
            )
            if differentiators:
                reusable.append(
                    "only if the new mechanism supplies: " + " | ".join(differentiators)
                )
            reframes = _unique_compact(audit.required_reframing, limit=2)
            if reframes:
                reusable.append("possible reframing: " + " | ".join(reframes))

        if is_selected:
            reasons.append(f"selected idea score was {report.selection.score}/10")
            if not ResearchFoundry.selection_gate_cleared(
                report.request, report.selection
            ):
                reasons.append("selected idea did not clear the full selector gate")
            if (
                selector_score_threshold is not None
                and report.selection.score < selector_score_threshold
            ):
                reasons.append(
                    f"selector score below retry target {selector_score_threshold}/10"
                )
            score_gaps = _selection_score_gaps(report)
            if score_gaps:
                reasons.append("selector score gaps: " + " | ".join(score_gaps))
            selector_risks = _unique_compact(report.selection.decisive_risks, limit=3)
            if selector_risks:
                reasons.append("selector risks: " + " | ".join(selector_risks))
            next_steps = _unique_compact(report.selection.required_next_steps, limit=3)
            if next_steps:
                reusable.append("required repair move: " + " | ".join(next_steps))
        else:
            reasons.append(
                "not selected over "
                + _compact_text(report.selection.selected_title, limit=120)
            )

        fatal_flaws = _unique_compact(
            [flaw for review in idea_reviews for flaw in review.fatal_flaws],
            limit=3,
        )
        if fatal_flaws:
            reasons.append("review fatal flaws: " + " | ".join(fatal_flaws))
            avoid.append("avoid unresolved fatal flaws: " + " | ".join(fatal_flaws))
        rescue_moves = _unique_compact(
            [move for review in idea_reviews for move in review.rescue_moves],
            limit=3,
        )
        if rescue_moves:
            reusable.append("reviewer rescue move: " + " | ".join(rescue_moves))

        if require_novelty_pass and not report_has_novelty_pass(report):
            reasons.append("batch had no novelty-audit pass")

        if idea.key_risks:
            avoid.append(
                "do not carry forward unresolved risks: "
                + " | ".join(_unique_compact(idea.key_risks, limit=3))
            )

        if not reasons:
            reasons.append(
                "batch failed another retry gate; do not carry forward unchanged"
            )
        if not avoid:
            avoid.append(
                "do not repeat, rename, merge, or lightly modify this core mechanism"
            )
        if not reusable:
            reusable.append(
                "none unless paired with a genuinely different mechanism that clears the failed gates"
            )

        lines.append(
            "   Why rejected: " + " | ".join(_unique_compact(reasons, limit=7))
        )
        lines.append("   Avoid next: " + " | ".join(_unique_compact(avoid, limit=4)))
        lines.append(
            "   Still useful: " + " | ".join(_unique_compact(reusable, limit=4))
        )

    return "\n".join(lines)


def retry_constraint(
    failed_reports: list[ResearchReport],
    attempt: int,
    max_batches: int,
    *,
    require_novelty_pass: bool,
    selector_score_threshold: int | None,
) -> str:
    prior_misses: list[str] = []
    selected_misses: list[str] = []
    for report in failed_reports:
        selected_misses.append(
            f"{report.selection.selected_title} [selector score {report.selection.score}/10]"
        )
        for audit in report.novelty_audits:
            blockers = "; ".join(audit.main_track_blockers[:2])
            collisions = "; ".join(
                collision.prior_work for collision in audit.closest_prior_work[:2]
            )
            if audit.main_track_verdict == "pass":
                reason = "novelty pass, but batch did not clear all retry gates"
            else:
                reason = blockers or collisions or "no defensible main-track novelty claim"
            prior_misses.append(f"{audit.idea_title} [{audit.main_track_verdict}: {reason}]")

    missed_text = " | ".join(prior_misses[-12:]) or "previous batch failed novelty audit"
    selected_text = " | ".join(selected_misses[-6:]) or "no previous selector decision"
    feedback_text = "\n\n".join(
        retry_feedback_summary(report) for report in failed_reports[-3:]
    )
    ledger_text = "\n\n".join(
        _rejected_method_ledger(
            report,
            batch_number=index,
            require_novelty_pass=require_novelty_pass,
            selector_score_threshold=selector_score_threshold,
        )
        for index, report in enumerate(failed_reports, start=1)
    )
    targets = retry_target_text(
        require_novelty_pass=require_novelty_pass,
        selector_score_threshold=selector_score_threshold,
    )
    return (
        f"Retry batch {attempt}/{max_batches}: generate a fresh idea set because previous "
        f"candidates did not clear {targets}. Previous selector decisions: {selected_text}. "
        "Do not repeat, rename, merge, or lightly modify these selected or audited "
        f"directions: {missed_text}. Search for a different core mechanism, a different "
        "decisive technical differentiator, and a stronger selector case.\n\n"
        "Rejected-method ledger from prior batches (binding retry memory):\n"
        f"{ledger_text}\n\n"
        "Reviewer feedback to use before generating the next batch:\n"
        f"{feedback_text}\n\n"
        "Before returning ideas, stress-test each candidate against this feedback. "
        "If a candidate mainly fixes the name, framing, or evaluation schedule while "
        "keeping the same weak mechanism, reject it internally and generate a stronger "
        "candidate."
    )


async def run_pipeline(args: argparse.Namespace) -> int:
    console = Console(stderr=True, highlight=True)
    settings = Settings()
    if args.out_dir:
        settings.output_dir = args.out_dir
    if args.max_wait_seconds is not None:
        settings.max_wait_seconds = args.max_wait_seconds

    selector_score_threshold = max(
        args.until_selector_score or args.ambition_floor,
        args.ambition_floor,
    )
    require_novelty_pass = args.until_novelty_pass > 1
    max_batches = max(args.until_novelty_pass, DEFAULT_SELECTOR_SCORE_BATCHES)
    base_request = build_request(args)
    gateway = DryRunGateway() if base_request.dry_run else OpenAIResponsesGateway(settings)
    store = RunStore(settings.output_dir)
    foundry = ResearchFoundry(gateway=gateway, settings=settings, store=store)
    failed_reports: list[ResearchReport] = []
    report: ResearchReport | None = None

    for attempt in range(1, max_batches + 1):
        retry_constraints = (
            [
                retry_constraint(
                    failed_reports,
                    attempt,
                    max_batches,
                    require_novelty_pass=require_novelty_pass,
                    selector_score_threshold=selector_score_threshold,
                )
            ]
            if failed_reports
            else None
        )
        request = build_request(args, retry_constraints=retry_constraints)
        if max_batches > 1:
            targets = retry_target_text(
                require_novelty_pass=require_novelty_pass,
                selector_score_threshold=selector_score_threshold,
            )
            console.print(
                Panel(
                    f"Batch {attempt}/{max_batches}: searching for {targets}.",
                    title="[bold bright_yellow]Retry-gated search[/]",
                    border_style="bright_yellow",
                )
            )

        progress = None if args.no_progress else RichTqdmReporter(console=console)
        report = await foundry.run(request, save=True, progress=progress)
        if report_clears_retry_gates(
            report,
            require_novelty_pass=require_novelty_pass,
            selector_score_threshold=selector_score_threshold,
        ):
            if max_batches > 1:
                console.print(
                    Panel(
                        f"Retry gates cleared in batch {attempt}/{max_batches}. "
                        "Stopping retry search.",
                        title="[bold green]Retry gates cleared[/]",
                        border_style="green",
                    )
                )
            break

        failed_reports.append(report)
        if attempt < max_batches:
            failure_text = "; ".join(
                retry_failure_reasons(
                    report,
                    require_novelty_pass=require_novelty_pass,
                    selector_score_threshold=selector_score_threshold,
                )
            )
            console.print(
                Panel(
                    f"{failure_text}. Starting a fresh research batch with the failed "
                    "directions excluded.",
                    title="[bold yellow]Retry gates not cleared[/]",
                    border_style="yellow",
                )
            )
        elif max_batches > 1:
            console.print(
                Panel(
                    f"Retry gates were not cleared after {max_batches} batches. The "
                    "latest report is saved, but it should be treated as a pivot "
                    "document, not a cleared main-paper plan.",
                    title="[bold red]Retry gates not cleared[/]",
                    border_style="red",
                )
            )

    if report is None:
        raise RuntimeError("No research batch was run.")

    if args.no_progress:
        run_dir = store.last_run_dir
        console.print("[bold green]Research Foundry complete.[/]")
        if run_dir:
            console.print(f"[cyan]Report:[/] {run_dir / 'report.md'}")
            console.print(f"[cyan]JSON:[/]   {run_dir / 'report.json'}")
            if report.implementation_docx_path:
                console.print(f"[cyan]DOCX:[/]   {report.implementation_docx_path}")
            else:
                console.print("[cyan]DOCX:[/]   skipped: selector gate not cleared")
        top = report.top_idea.title if report.top_idea else "No parsed top idea"
        console.print(f"[bold cyan]Top idea:[/] {top}")
        console.print_json(ResearchFoundry.report_summary(report))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            raise SystemExit(asyncio.run(run_pipeline(args)))
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        except Exception as exc:
            Console(stderr=True).print(f"[bold red]error:[/] {exc}")
            raise SystemExit(1) from None

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
