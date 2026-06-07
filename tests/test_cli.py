import json

import pytest

from research_foundry import cli
from research_foundry.gateway import DryRunGateway


def test_until_novelty_pass_flag_defaults_to_one_batch():
    args = cli.build_parser().parse_args(
        ["run", "--field", "prompt optimization", "--objective", "find ideas"]
    )

    assert args.until_novelty_pass == 1


def test_until_novelty_pass_flag_without_value_uses_default_budget():
    args = cli.build_parser().parse_args(
        [
            "run",
            "--field",
            "prompt optimization",
            "--objective",
            "find ideas",
            "--until-novelty-pass",
        ]
    )

    assert args.until_novelty_pass == cli.DEFAULT_NOVELTY_PASS_BATCHES


def test_until_novelty_pass_flag_accepts_explicit_budget():
    args = cli.build_parser().parse_args(
        [
            "run",
            "--field",
            "prompt optimization",
            "--objective",
            "find ideas",
            "--until-novelty-pass",
            "5",
        ]
    )

    assert args.until_novelty_pass == 5


def test_until_selector_score_flag_defaults_to_disabled():
    args = cli.build_parser().parse_args(
        ["run", "--field", "prompt optimization", "--objective", "find ideas"]
    )

    assert args.until_selector_score is None


def test_until_selector_score_flag_without_value_uses_default_threshold():
    args = cli.build_parser().parse_args(
        [
            "run",
            "--field",
            "prompt optimization",
            "--objective",
            "find ideas",
            "--until-selector-score",
        ]
    )

    assert args.until_selector_score == cli.DEFAULT_SELECTOR_SCORE_THRESHOLD


def test_until_selector_score_flag_accepts_explicit_threshold():
    args = cli.build_parser().parse_args(
        [
            "run",
            "--field",
            "prompt optimization",
            "--objective",
            "find ideas",
            "--until-selector-score",
            "9",
        ]
    )

    assert args.until_selector_score == 9


class _FailThenPassGateway(DryRunGateway):
    def __init__(self):
        self.audit_calls = 0

    def _content_for(self, output_kind, agent_name):
        if output_kind != "novelty_audit":
            return super()._content_for(output_kind, agent_name)

        self.audit_calls += 1
        if self.audit_calls > 1:
            return super()._content_for(output_kind, agent_name)

        return json.dumps(
            {
                "audits": [
                    {
                        "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                        "closest_prior_work": [],
                        "novelty_score": 5,
                        "paper_worth_score": 5,
                        "venue_upside_score": 5,
                        "main_track_verdict": "fail",
                        "workshop_risk": True,
                        "main_track_blockers": ["Too close to prior reviewer benchmarks."],
                        "required_reframing": ["Find a different core mechanism."],
                        "direct_comparisons_required": [],
                    },
                    {
                        "idea_title": "Citation-Causal Idea Search for ML Paper Generation",
                        "closest_prior_work": [],
                        "novelty_score": 6,
                        "paper_worth_score": 6,
                        "venue_upside_score": 6,
                        "main_track_verdict": "borderline",
                        "workshop_risk": True,
                        "main_track_blockers": ["Too close to retrieval ideation."],
                        "required_reframing": ["Narrow to a new formal target."],
                        "direct_comparisons_required": [],
                    },
                ],
                "batch_verdict": "fail",
                "batch_rationale": "No idea clears the main-track novelty bar.",
            }
        )


@pytest.mark.asyncio
async def test_until_novelty_pass_retries_until_a_batch_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DryRunGateway", _FailThenPassGateway)
    args = cli.build_parser().parse_args(
        [
            "run",
            "--field",
            "prompt optimization",
            "--objective",
            "find ideas",
            "--dry-run",
            "--no-progress",
            "--out-dir",
            str(tmp_path),
            "--until-novelty-pass",
            "2",
        ]
    )

    exit_code = await cli.run_pipeline(args)

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert exit_code == 0
    assert len(run_dirs) == 2

    first_report = json.loads((run_dirs[0] / "report.json").read_text())
    second_report = json.loads((run_dirs[1] / "report.json").read_text())
    assert all(
        audit["main_track_verdict"] != "pass"
        for audit in first_report["novelty_audits"]
    )
    assert any(
        audit["main_track_verdict"] == "pass"
        for audit in second_report["novelty_audits"]
    )
    assert "Retry batch 2/2" in second_report["request"]["constraints"][-1]


class _LowThenHighSelectionGateway(DryRunGateway):
    def __init__(self):
        self.selection_calls = 0

    def _content_for(self, output_kind, agent_name):
        if output_kind != "selection":
            return super()._content_for(output_kind, agent_name)

        self.selection_calls += 1
        if self.selection_calls > 1:
            return super()._content_for(output_kind, agent_name)

        return json.dumps(
            {
                "selected_title": "Novelty Stress Tests for Scientific Agent Systems",
                "rationale": "Promising but not yet a strong enough selector decision.",
                "iclr_neurips_case": "Needs a sharper paper case.",
                "research_worth_score": 8,
                "paper_worth_score": 8,
                "venue_upside_score": 8,
                "fixed_pool_only_score": 8,
                "breakthrough_condition": "Needs stronger expert-label evidence.",
                "decisive_strengths": ["Timely evaluation target"],
                "decisive_risks": ["Selector confidence is below threshold"],
                "required_next_steps": ["Generate a stronger idea batch"],
                "score": 7,
            }
        )


@pytest.mark.asyncio
async def test_until_selector_score_retries_until_score_clears_threshold(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "DryRunGateway", _LowThenHighSelectionGateway)
    args = cli.build_parser().parse_args(
        [
            "run",
            "--field",
            "prompt optimization",
            "--objective",
            "find ideas",
            "--dry-run",
            "--no-progress",
            "--out-dir",
            str(tmp_path),
            "--until-selector-score",
        ]
    )

    exit_code = await cli.run_pipeline(args)

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert exit_code == 0
    assert len(run_dirs) == 2

    first_report = json.loads((run_dirs[0] / "report.json").read_text())
    second_report = json.loads((run_dirs[1] / "report.json").read_text())
    assert first_report["selection"]["score"] == 7
    assert second_report["selection"]["score"] == 8
    retry_feedback = second_report["request"]["constraints"][-1]
    assert "Retry batch 2/3" in retry_feedback
    assert "selector score 7/10" in retry_feedback
    assert "Reviewer feedback to use before generating the next batch" in retry_feedback
    assert "Selector risks to fix: Selector confidence is below threshold" in retry_feedback
    assert "Required repair moves: Generate a stronger idea batch" in retry_feedback
    assert "do not merely avoid the old titles" in retry_feedback
