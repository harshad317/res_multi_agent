from __future__ import annotations

from research_foundry.models import ResearchReport


def render_report_markdown(report: ResearchReport) -> str:
    lines: list[str] = []
    lines.append("# Research Foundry Report")
    lines.append("")
    lines.append(f"Generated: {report.generated_at.isoformat()}")
    lines.append("")
    lines.append("## Request")
    lines.append("")
    lines.append(f"- Field: {report.request.field}")
    lines.append(f"- Objective: {report.request.objective}")
    lines.append(f"- Venues: {', '.join(report.request.target_venues)}")
    lines.append(f"- Ideas requested: {report.request.idea_count}")
    lines.append(f"- Ambition floor: {report.request.ambition_floor}/10")
    if report.request.constraints:
        lines.append("- Constraints:")
        for constraint in report.request.constraints:
            lines.append(f"  - {constraint}")
    lines.append("")

    lines.append("## Top Idea")
    lines.append("")
    top = report.top_idea
    if top:
        lines.append(f"**{top.title}**")
        lines.append("")
        lines.append(top.thesis)
        lines.append("")
        lines.append(f"Selector score: {report.selection.score}/10")
        score_parts = [
            ("research worth", report.selection.research_worth_score),
            ("paper worth", report.selection.paper_worth_score),
            ("venue upside", report.selection.venue_upside_score),
            ("fixed-pool only", report.selection.fixed_pool_only_score),
        ]
        available_scores = [
            f"{label}: {value}/10" for label, value in score_parts if value is not None
        ]
        if available_scores:
            lines.append(f"Upside scores: {', '.join(available_scores)}")
        if report.selection.breakthrough_condition:
            lines.append(f"Breakthrough condition: {report.selection.breakthrough_condition}")
        selected_audit = _audit_for_title(report, top.title)
        if selected_audit:
            lines.append(
                "Novelty audit: "
                f"{selected_audit.main_track_verdict} "
                f"(novelty {selected_audit.novelty_score}/10, "
                f"paper {selected_audit.paper_worth_score}/10, "
                f"venue {selected_audit.venue_upside_score}/10)"
            )
        if report.selection.main_conference_checklist:
            lines.append("")
            lines.append("Main conference checklist:")
            lines.append("")
            lines.append("| Check | Score | Evidence Needed | Failure Mode |")
            lines.append("| --- | ---: | --- | --- |")
            for check in report.selection.main_conference_checklist:
                lines.append(
                    f"| {check.name} | {check.score}/{check.threshold} | "
                    f"{check.evidence_needed} | {check.failure_mode} |"
                )
        lines.append("")
        lines.append(report.selection.rationale)
    else:
        lines.append("No top idea parsed.")
    lines.append("")

    if report.implementation_docx_path:
        lines.append(f"Implementation DOCX: `{report.implementation_docx_path}`")
    lines.append("")

    if report.novelty_audits:
        lines.append("## Novelty Collision Audit")
        lines.append("")
        lines.append(
            "| Idea | Verdict | Novelty | Paper | Venue | Workshop Risk | Closest Collisions | Blockers |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
        for audit in report.novelty_audits:
            collisions = "; ".join(
                f"{collision.prior_work} ({collision.severity})"
                for collision in audit.closest_prior_work
            ) or "-"
            blockers = "; ".join(audit.main_track_blockers) or "-"
            lines.append(
                f"| {audit.idea_title} | {audit.main_track_verdict} | "
                f"{audit.novelty_score}/10 | {audit.paper_worth_score}/10 | "
                f"{audit.venue_upside_score}/10 | {audit.workshop_risk} | "
                f"{collisions} | {blockers} |"
            )
        lines.append("")
        lines.append("### Required Reframing And Direct Comparisons")
        lines.append("")
        for audit in report.novelty_audits:
            lines.append(f"#### {audit.idea_title}")
            if audit.required_reframing:
                lines.append("")
                lines.append("Required reframing:")
                for item in audit.required_reframing:
                    lines.append(f"- {item}")
            if audit.direct_comparisons_required:
                lines.append("")
                lines.append("Direct comparisons required:")
                for item in audit.direct_comparisons_required:
                    lines.append(f"- {item}")
            lines.append("")

    lines.append("## Ideas")
    lines.append("")
    for idx, idea in enumerate(report.ideas, start=1):
        lines.append(f"### {idx}. {idea.title}")
        lines.append("")
        lines.append(f"**Thesis:** {idea.thesis}")
        lines.append("")
        lines.append(f"**Mechanism:** {idea.core_mechanism}")
        lines.append("")
        lines.append(f"**Novelty claim:** {idea.novelty_claim}")
        lines.append("")
        if idea.decisive_differentiator:
            lines.append(f"**Decisive differentiator:** {idea.decisive_differentiator}")
            lines.append("")
        idea_scores = [
            ("innovation", idea.innovation_score),
            ("paper worth", idea.paper_worth_score),
            ("venue upside", idea.venue_upside_score),
            ("fixed-pool only", idea.fixed_pool_only_score),
        ]
        idea_score_text = [
            f"{label}: {value}/10" for label, value in idea_scores if value is not None
        ]
        if idea_score_text:
            lines.append(f"**Idea scores:** {', '.join(idea_score_text)}")
            lines.append("")
        if idea.main_conference_checklist:
            lines.append("**Main conference checklist:**")
            lines.append("")
            lines.append("| Check | Score | Evidence Needed | Failure Mode |")
            lines.append("| --- | ---: | --- | --- |")
            for check in idea.main_conference_checklist:
                lines.append(
                    f"| {check.name} | {check.score}/{check.threshold} | "
                    f"{check.evidence_needed} | {check.failure_mode} |"
                )
            lines.append("")
        if idea.nearest_prior_work:
            lines.append("**Nearest prior work:**")
            for prior in idea.nearest_prior_work:
                lines.append(f"- {prior}")
            lines.append("")
        if idea.first_experiments:
            lines.append("**First experiments:**")
            for experiment in idea.first_experiments:
                lines.append(f"- {experiment}")
            lines.append("")
        if idea.key_risks:
            lines.append("**Risks:**")
            for risk in idea.key_risks:
                lines.append(f"- {risk}")
            lines.append("")

    lines.append("## Review Scores")
    lines.append("")
    if report.reviews:
        lines.append(
            "| Idea | Reviewer | Avg | Research | Paper | Venue | Fixed-Pool | Recommendation | Fatal Flaws |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for review in report.reviews:
            flaws = "; ".join(review.fatal_flaws) if review.fatal_flaws else "-"
            research = _score_cell(review.research_worth_score)
            paper = _score_cell(review.paper_worth_score)
            venue = _score_cell(review.venue_upside_score)
            fixed_pool = _score_cell(review.fixed_pool_only_score)
            lines.append(
                f"| {review.idea_title} | {review.reviewer} | {review.average} | "
                f"{research} | {paper} | {venue} | {fixed_pool} | "
                f"{review.recommendation} | {flaws} |"
            )
    else:
        lines.append("No structured reviews parsed.")
    lines.append("")

    lines.append("## Experiment Plan")
    lines.append("")
    lines.append(report.experiment_plan.content)
    lines.append("")

    lines.append("## Detailed Implementation Plan")
    lines.append("")
    lines.append(report.implementation_plan.content)
    lines.append("")

    lines.append("## Final Recommendation")
    lines.append("")
    lines.append(report.final_recommendation.content)
    lines.append("")

    if report.literature.sources:
        lines.append("## Literature Sources")
        lines.append("")
        for source in report.literature.sources:
            title = source.title or source.url
            lines.append(f"- [{title}]({source.url})")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _score_cell(value: int | None) -> str:
    return "-" if value is None else str(value)


def _audit_for_title(report: ResearchReport, title: str):
    for audit in report.novelty_audits:
        if audit.idea_title == title:
            return audit
    return None
