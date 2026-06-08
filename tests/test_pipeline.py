import json

import pytest

from research_foundry.config import Settings
from research_foundry.gateway import DryRunGateway
from research_foundry.models import (
    AgentArtifact,
    ConferenceCheck,
    IdeaCandidate,
    IdeaNoveltyAudit,
    NoveltyCollision,
    ResearchRequest,
    SelectionDecision,
)
from research_foundry.pipeline import ResearchFoundry
from research_foundry.prompts import idea_generation_prompt, selection_prompt
from research_foundry.storage import RunStore


class RecordingGateway(DryRunGateway):
    def __init__(self):
        self.calls = []

    async def run_text(self, **kwargs):
        self.calls.append(kwargs)
        return await super().run_text(**kwargs)


class RecordingProgress:
    def __init__(self):
        self.events = []

    def start(self, request, total_stages):
        self.events.append(("start", request.field, total_stages))

    def stage_start(self, index, total, name, detail):
        self.events.append(("stage_start", index, total, name, detail))

    def stage_tick(self, index, total, name):
        self.events.append(("stage_tick", index, total, name))

    def stage_complete(self, index, total, name, artifact):
        self.events.append(("stage_complete", index, total, name, artifact.agent_name))

    def stage_error(self, index, total, name, error):
        self.events.append(("stage_error", index, total, name, str(error)))

    def ideas_ready(self, ideas):
        self.events.append(("ideas_ready", [idea.title for idea in ideas]))

    def novelty_audit_ready(self, audits):
        self.events.append(
            ("novelty_audit_ready", [(audit.idea_title, audit.main_track_verdict) for audit in audits])
        )

    def selection_ready(self, selection, selected_idea):
        self.events.append(("selection_ready", selection.selected_title, selected_idea.title))

    def saved(self, report, run_dir):
        self.events.append(("saved", report.selection.selected_title, run_dir))

    def finish(self, report):
        self.events.append(("finish", report.selection.selected_title))


@pytest.mark.asyncio
async def test_dry_run_pipeline_returns_structured_report(tmp_path):
    request = ResearchRequest(
        field="multi-agent systems for scientific discovery",
        objective="generate original ICLR/NeurIPS ideas",
        constraints=["must have executable evaluations"],
        dry_run=True,
    )
    settings = Settings(output_dir=str(tmp_path))
    foundry = ResearchFoundry(
        gateway=DryRunGateway(),
        settings=settings,
        store=RunStore(tmp_path),
    )

    report = await foundry.run(request)

    assert report.ideas
    assert report.reviews
    assert report.selection.selected_title
    assert report.experiment_plan.content
    assert report.implementation_plan.content
    assert report.implementation_docx_path
    assert report.top_idea is not None
    run_dir = tmp_path / next(tmp_path.iterdir()).name
    assert (run_dir / "report.md").exists()
    assert (run_dir / "selected_idea_implementation_plan.docx").exists()


@pytest.mark.asyncio
async def test_all_agent_stages_receive_web_search_tool(tmp_path):
    request = ResearchRequest(
        field="prompt optimization",
        objective="beat GEPA and MIPROv2 cost-normalized performance",
        dry_run=True,
    )
    gateway = RecordingGateway()
    foundry = ResearchFoundry(
        gateway=gateway,
        settings=Settings(output_dir=str(tmp_path)),
        store=RunStore(tmp_path),
    )

    await foundry.run(request)

    assert {call["agent_name"] for call in gateway.calls} == {
        "Literature Cartographer",
        "Novelty Gap Miner",
        "Idea Generator",
        "Novelty Collision Auditor",
        "Novelty Score Auditor",
        "Paper Worth Score Auditor",
        "Venue Upside Score Auditor",
        "Skeptical Review Board",
        "Best Idea Selector",
        "Experiment Designer",
        "Implementation Architect",
        "Chief Scientist",
    }
    for call in gateway.calls:
        tools = call.get("tools") or []
        assert any(tool.get("type") == "web_search" for tool in tools)
        if call["model"] in {"gpt-5.5", "gpt-5.5-pro"}:
            assert call.get("reasoning_effort") == "high"


class IndependentNoveltyScoreGateway(DryRunGateway):
    def _content_for(self, output_kind, agent_name):
        if output_kind == "novelty_audit":
            return json.dumps(
                {
                    "audits": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "closest_prior_work": [],
                            "novelty_score": 1,
                            "paper_worth_score": 1,
                            "venue_upside_score": 1,
                            "main_track_verdict": "pass",
                            "workshop_risk": False,
                            "main_track_blockers": [],
                            "required_reframing": [],
                            "direct_comparisons_required": [],
                        }
                    ],
                    "batch_verdict": "pass",
                    "batch_rationale": "Testing score overwrite.",
                }
            )
        if output_kind == "novelty_score_audit":
            return json.dumps(
                {
                    "scores": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "score": 8,
                            "rationale": "Novelty scorer value.",
                            "risks": [],
                        }
                    ]
                }
            )
        if output_kind == "paper_worth_score_audit":
            return json.dumps(
                {
                    "scores": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "score": 7,
                            "rationale": "Paper scorer value.",
                            "risks": [],
                        }
                    ]
                }
            )
        if output_kind == "venue_upside_score_audit":
            return json.dumps(
                {
                    "scores": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "score": 9,
                            "rationale": "Venue scorer value.",
                            "risks": [],
                        }
                    ]
                }
            )
        return super()._content_for(output_kind, agent_name)


class RejectedSelectionGateway(RecordingGateway):
    def _content_for(self, output_kind, agent_name):
        if output_kind == "selection":
            return json.dumps(
                {
                    "selected_title": "Novelty Stress Tests for Scientific Agent Systems",
                    "rationale": (
                        "Batch failed ambition floor; this is only the strongest "
                        "salvage direction."
                    ),
                    "iclr_neurips_case": "Not strong enough to justify implementation.",
                    "research_worth_score": 9,
                    "paper_worth_score": 9,
                    "venue_upside_score": 9,
                    "fixed_pool_only_score": 8,
                    "breakthrough_condition": "Needs a stronger fixed-pool MVP.",
                    "decisive_strengths": ["Interesting diagnostic direction"],
                    "decisive_risks": ["fixed_pool_only_score is below the floor"],
                    "required_next_steps": ["Generate a stronger idea batch"],
                    "score": 6,
                }
            )
        return super()._content_for(output_kind, agent_name)


@pytest.mark.asyncio
async def test_rejected_selection_skips_downstream_agents(tmp_path):
    request = ResearchRequest(
        field="prompt optimization",
        objective="find breakthrough ideas",
        ambition_floor=9,
        dry_run=True,
    )
    gateway = RejectedSelectionGateway()
    foundry = ResearchFoundry(
        gateway=gateway,
        settings=Settings(output_dir=str(tmp_path)),
        store=RunStore(tmp_path),
    )

    report = await foundry.run(request)

    called_agents = {call["agent_name"] for call in gateway.calls}
    assert "Best Idea Selector" in called_agents
    assert "Experiment Designer" not in called_agents
    assert "Implementation Architect" not in called_agents
    assert "Chief Scientist" not in called_agents
    assert report.selection.score == 6
    assert "Gate status: not cleared" in report.experiment_plan.content
    assert "Required gate: selector score" in report.experiment_plan.content
    assert report.experiment_plan.metadata["skipped_due_to_selector_gate"] is True
    assert report.experiment_plan.metadata["selector_gate_cleared"] is False
    assert report.implementation_plan.metadata["skipped_due_to_selector_gate"] is True
    assert report.final_recommendation.metadata["skipped_due_to_selector_gate"] is True
    assert report.implementation_docx_path is None
    run_dir = tmp_path / next(tmp_path.iterdir()).name
    assert not (run_dir / "selected_idea_implementation_plan.docx").exists()


@pytest.mark.asyncio
async def test_novelty_audit_scores_come_from_independent_subagents(tmp_path):
    request = ResearchRequest(
        field="prompt optimization",
        objective="find breakthrough ideas",
        dry_run=True,
    )
    foundry = ResearchFoundry(
        gateway=IndependentNoveltyScoreGateway(),
        settings=Settings(output_dir=str(tmp_path)),
        store=RunStore(tmp_path),
    )

    report = await foundry.run(request)

    audit = report.novelty_audits[0]
    assert audit.novelty_score == 8
    assert audit.paper_worth_score == 7
    assert audit.venue_upside_score == 9


@pytest.mark.asyncio
async def test_pipeline_emits_progress_events(tmp_path):
    request = ResearchRequest(
        field="prompt optimization",
        objective="beat GEPA and MIPROv2 cost-normalized performance",
        dry_run=True,
    )
    progress = RecordingProgress()
    foundry = ResearchFoundry(
        gateway=DryRunGateway(),
        settings=Settings(output_dir=str(tmp_path)),
        store=RunStore(tmp_path),
    )

    await foundry.run(request, progress=progress)

    stage_starts = [event for event in progress.events if event[0] == "stage_start"]
    stage_completes = [event for event in progress.events if event[0] == "stage_complete"]
    assert progress.events[0] == ("start", "prompt optimization", 9)
    assert len(stage_starts) == 9
    assert len(stage_completes) == 9
    assert stage_starts[0][3] == "Literature Cartographer"
    assert stage_completes[-1][3] == "Chief Scientist"
    assert any(event[0] == "ideas_ready" for event in progress.events)
    assert any(event[0] == "novelty_audit_ready" for event in progress.events)
    assert any(event[0] == "selection_ready" for event in progress.events)
    assert any(event[0] == "saved" for event in progress.events)
    assert progress.events[-1][0] == "finish"


def test_ambition_gate_marks_weak_selection_below_floor():
    request = ResearchRequest(
        field="prompt optimization",
        objective="find breakthrough ideas",
        ambition_floor=8,
    )
    selection = SelectionDecision(
        selected_title="Idea",
        rationale="Promising but weak.",
        iclr_neurips_case="Maybe.",
        research_worth_score=8,
        paper_worth_score=8,
        venue_upside_score=8,
        fixed_pool_only_score=8,
        main_conference_checklist=[
            ConferenceCheck(
                name="Strong baseline and SOTA comparison plan",
                score=7,
                threshold=8,
                evidence_needed="Head-to-head comparison against current SOTA.",
                failure_mode="Reviewers call it undercompared.",
            )
        ],
        score=9,
    )

    ResearchFoundry._enforce_ambition_gate(request, selection)

    assert selection.score == 7
    assert "checklist: Strong baseline" in selection.decisive_risks[0]
    assert "Strong baseline and SOTA comparison plan=7/10" in selection.decisive_risks[0]
    assert selection.required_next_steps


def test_novelty_audit_gate_caps_failed_selection():
    request = ResearchRequest(
        field="prompt optimization",
        objective="find breakthrough ideas",
        ambition_floor=8,
    )
    selection = SelectionDecision(
        selected_title="Idea",
        rationale="Looks strong before audit.",
        iclr_neurips_case="Maybe.",
        research_worth_score=9,
        paper_worth_score=9,
        venue_upside_score=9,
        fixed_pool_only_score=9,
        score=9,
    )
    audits = [
        IdeaNoveltyAudit(
            idea_title="Idea",
            closest_prior_work=[
                NoveltyCollision(
                    prior_work="Closest Prior",
                    overlap="Already covers the core mechanism.",
                    severity="fatal",
                    required_differentiator="Show a different formal object.",
                )
            ],
            novelty_score=5,
            paper_worth_score=5,
            venue_upside_score=5,
            main_track_verdict="fail",
            workshop_risk=True,
            main_track_blockers=["Headline already owned by prior work."],
            direct_comparisons_required=["Closest Prior"],
        )
    ]

    ResearchFoundry._enforce_novelty_audit_gate(request, selection, audits)

    assert selection.score == 5
    assert "Novelty collision gate not cleared" in selection.decisive_risks[0]
    assert any("Closest Prior" in step for step in selection.required_next_steps)


def test_novelty_audit_gate_does_not_cap_pass_with_high_collision():
    request = ResearchRequest(
        field="prompt optimization",
        objective="find breakthrough ideas",
        ambition_floor=8,
    )
    selection = SelectionDecision(
        selected_title="Idea",
        rationale="Strong but crowded.",
        iclr_neurips_case="Clear differentiator remains.",
        research_worth_score=9,
        paper_worth_score=9,
        venue_upside_score=9,
        fixed_pool_only_score=8,
        score=9,
    )
    audits = [
        IdeaNoveltyAudit(
            idea_title="Idea",
            closest_prior_work=[
                NoveltyCollision(
                    prior_work="High Collision Prior",
                    overlap="Close but not fatal.",
                    severity="high",
                    required_differentiator="Show a different formal object.",
                )
            ],
            novelty_score=8,
            paper_worth_score=9,
            venue_upside_score=9,
            main_track_verdict="pass",
            workshop_risk=False,
            main_track_blockers=["Must not look like prior work plus engineering."],
            direct_comparisons_required=["High Collision Prior"],
        )
    ]

    ResearchFoundry._enforce_novelty_audit_gate(request, selection, audits)

    assert selection.score == 9
    assert "High novelty collisions remain" in selection.decisive_risks[0]
    assert any("Resolve novelty blocker" in step for step in selection.required_next_steps)
    assert any("High Collision Prior" in step for step in selection.required_next_steps)


def test_invalid_novelty_audit_fails_closed():
    foundry = ResearchFoundry(gateway=DryRunGateway())
    artifact = AgentArtifact(
        agent_name="Novelty Collision Auditor",
        model="gpt-5.5-pro",
        content="not valid json",
    )
    ideas = [
        IdeaCandidate(
            title="Idea A",
            thesis="A promising method.",
            core_mechanism="Mechanism.",
            novelty_claim="Claim.",
            why_reviewers_care="Reason.",
        )
    ]

    audits = foundry._parse_novelty_audits(artifact, ideas)

    assert len(audits) == 1
    assert audits[0].idea_title == "Idea A"
    assert audits[0].main_track_verdict == "fail"
    assert audits[0].novelty_score == 1
    assert audits[0].workshop_risk is True
    assert "invalid" in audits[0].main_track_blockers[0]


def test_idea_generation_prompt_treats_retry_feedback_as_binding():
    request = ResearchRequest(
        field="prompt optimization",
        objective="find breakthrough ideas",
        constraints=[
            "Failed batch feedback: selector score 7/10; weak fixed-pool MVP."
        ],
        ambition_floor=8,
    )

    prompt = idea_generation_prompt(request, "Prior work map.", [])

    assert "Retry-feedback handling" in prompt
    assert "treat it as binding" in prompt
    assert "rejected-method ledger" in prompt
    assert "Respect hard exclusions" in prompt
    assert "would top out" in prompt
    assert "7/10" in prompt


def test_selection_prompt_does_not_collapse_failed_gates_to_six():
    request = ResearchRequest(
        field="prompt optimization",
        objective="find breakthrough ideas",
        ambition_floor=9,
    )
    ideas = [
        IdeaCandidate(
            title="Strong Near Miss",
            thesis="A high-upside but incomplete paper direction.",
            core_mechanism="Mechanism.",
            novelty_claim="Claim.",
            why_reviewers_care="Reason.",
        )
    ]

    prompt = selection_prompt(request, "Prior work map.", ideas, [], "[]")

    assert "Do not collapse every failed-gate selection to 6/10" in prompt
    assert "set score <= 6" not in prompt
    assert "score must" in prompt
    assert "stay below the ambition floor" in prompt
