from __future__ import annotations

import re
from pathlib import Path

from research_foundry.docx_writer import write_implementation_docx
from research_foundry.models import AgentArtifact, ResearchReport
from research_foundry.render import render_report_markdown


class RunStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.last_run_dir: Path | None = None

    def save(self, report: ResearchReport) -> Path:
        run_id = self._run_id(report)
        run_dir = self.output_dir / run_id
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if not report.implementation_plan.metadata.get("skipped_due_to_selector_gate"):
            docx_path = run_dir / "selected_idea_implementation_plan.docx"
            write_implementation_docx(report, docx_path)
            report.implementation_docx_path = str(docx_path)
        else:
            report.implementation_docx_path = None

        (run_dir / "report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "report.md").write_text(render_report_markdown(report), encoding="utf-8")

        for artifact in [report.literature, *report.artifacts, report.final_recommendation]:
            self._write_artifact(artifacts_dir, artifact)

        self.last_run_dir = run_dir
        return run_dir

    def _write_artifact(self, artifacts_dir: Path, artifact: AgentArtifact) -> None:
        filename = self._slug(artifact.agent_name) + ".md"
        body = [
            f"# {artifact.agent_name}",
            "",
            f"Model: `{artifact.model}`",
            "",
            artifact.content,
            "",
        ]
        if artifact.sources:
            body.append("## Sources")
            body.append("")
            for source in artifact.sources:
                title = source.title or source.url
                body.append(f"- [{title}]({source.url})")
        (artifacts_dir / filename).write_text("\n".join(body).strip() + "\n", encoding="utf-8")

    def _run_id(self, report: ResearchReport) -> str:
        timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S_%f")
        return f"{timestamp}_{self._slug(report.request.field)[:48]}"

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
        return slug or "run"
