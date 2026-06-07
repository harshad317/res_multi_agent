from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    title: str | None = None
    url: str
    note: str | None = None


class ConferenceCheck(BaseModel):
    name: str
    score: int = Field(ge=1, le=10)
    threshold: int = Field(default=8, ge=1, le=10)
    evidence_needed: str
    failure_mode: str

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


class AgentArtifact(BaseModel):
    agent_name: str
    model: str
    content: str
    sources: list[SourceRef] = Field(default_factory=list)
    response_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchRequest(BaseModel):
    field: str
    objective: str
    constraints: list[str] = Field(default_factory=list)
    target_venues: list[str] = Field(default_factory=lambda: ["ICLR", "NeurIPS"])
    idea_count: int = Field(default=3, ge=1, le=8)
    risk_tolerance: Literal["conservative", "balanced", "aggressive"] = "balanced"
    ambition_floor: int = Field(default=8, ge=1, le=10)
    use_deep_research: bool = True
    fast: bool = False
    dry_run: bool = False


class NoveltyGap(BaseModel):
    gap: str
    why_now: str
    nearest_prior_work: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"] = "medium"


class IdeaCandidate(BaseModel):
    title: str
    thesis: str
    core_mechanism: str
    novelty_claim: str
    why_reviewers_care: str
    nearest_prior_work: list[str] = Field(default_factory=list)
    decisive_differentiator: str = ""
    innovation_score: int | None = Field(default=None, ge=1, le=10)
    paper_worth_score: int | None = Field(default=None, ge=1, le=10)
    venue_upside_score: int | None = Field(default=None, ge=1, le=10)
    fixed_pool_only_score: int | None = Field(default=None, ge=1, le=10)
    main_conference_checklist: list[ConferenceCheck] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    first_experiments: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    citations: list[SourceRef] = Field(default_factory=list)


class ReviewScore(BaseModel):
    idea_title: str
    reviewer: str
    novelty: int = Field(ge=1, le=10)
    significance: int = Field(ge=1, le=10)
    correctness: int = Field(ge=1, le=10)
    feasibility: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    research_worth_score: int | None = Field(default=None, ge=1, le=10)
    paper_worth_score: int | None = Field(default=None, ge=1, le=10)
    venue_upside_score: int | None = Field(default=None, ge=1, le=10)
    fixed_pool_only_score: int | None = Field(default=None, ge=1, le=10)
    main_conference_checklist: list[ConferenceCheck] = Field(default_factory=list)
    recommendation: Literal["strong_reject", "reject", "borderline", "accept", "strong_accept"]
    fatal_flaws: list[str] = Field(default_factory=list)
    required_experiments: list[str] = Field(default_factory=list)
    rescue_moves: list[str] = Field(default_factory=list)

    @property
    def average(self) -> float:
        return round(
            (
                self.novelty
                + self.significance
                + self.correctness
                + self.feasibility
                + self.clarity
            )
            / 5,
            2,
        )


class NoveltyCollision(BaseModel):
    prior_work: str
    url: str | None = None
    overlap: str
    severity: Literal["low", "medium", "high", "fatal"] = "medium"
    what_to_cede: str = ""
    required_differentiator: str = ""


class IdeaNoveltyAudit(BaseModel):
    idea_title: str
    closest_prior_work: list[NoveltyCollision] = Field(default_factory=list)
    novelty_score: int = Field(ge=1, le=10)
    paper_worth_score: int = Field(ge=1, le=10)
    venue_upside_score: int = Field(ge=1, le=10)
    main_track_verdict: Literal["pass", "borderline", "fail"]
    workshop_risk: bool = False
    main_track_blockers: list[str] = Field(default_factory=list)
    required_reframing: list[str] = Field(default_factory=list)
    direct_comparisons_required: list[str] = Field(default_factory=list)


class SelectionDecision(BaseModel):
    selected_title: str
    rationale: str
    iclr_neurips_case: str
    research_worth_score: int | None = Field(default=None, ge=1, le=10)
    paper_worth_score: int | None = Field(default=None, ge=1, le=10)
    venue_upside_score: int | None = Field(default=None, ge=1, le=10)
    fixed_pool_only_score: int | None = Field(default=None, ge=1, le=10)
    main_conference_checklist: list[ConferenceCheck] = Field(default_factory=list)
    breakthrough_condition: str = ""
    decisive_strengths: list[str] = Field(default_factory=list)
    decisive_risks: list[str] = Field(default_factory=list)
    required_next_steps: list[str] = Field(default_factory=list)
    score: int = Field(ge=1, le=10)


class ResearchReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: ResearchRequest
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    literature: AgentArtifact
    novelty_gaps: list[NoveltyGap]
    ideas: list[IdeaCandidate]
    novelty_audits: list[IdeaNoveltyAudit] = Field(default_factory=list)
    reviews: list[ReviewScore]
    selection: SelectionDecision
    experiment_plan: AgentArtifact
    implementation_plan: AgentArtifact
    final_recommendation: AgentArtifact
    implementation_docx_path: str | None = None
    artifacts: list[AgentArtifact] = Field(default_factory=list)

    @property
    def top_idea(self) -> IdeaCandidate | None:
        if not self.ideas:
            return None
        if self.selection:
            for idea in self.ideas:
                if idea.title == self.selection.selected_title:
                    return idea
        if not self.reviews:
            return self.ideas[0]

        scores: dict[str, list[float]] = {}
        for review in self.reviews:
            scores.setdefault(review.idea_title, []).append(review.average)

        def idea_score(idea: IdeaCandidate) -> float:
            values = scores.get(idea.title, [])
            return sum(values) / len(values) if values else 0.0

        return max(self.ideas, key=idea_score)
