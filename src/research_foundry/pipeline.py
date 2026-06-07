from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Protocol

from research_foundry.config import Settings
from research_foundry.gateway import ModelGateway, web_search_tool
from research_foundry.models import (
    AgentArtifact,
    IdeaCandidate,
    IdeaNoveltyAudit,
    NoveltyGap,
    ResearchReport,
    ResearchRequest,
    ReviewScore,
    SelectionDecision,
    SourceRef,
)
from research_foundry.prompts import (
    experiment_design_prompt,
    final_synthesis_prompt,
    implementation_doc_prompt,
    idea_generation_prompt,
    literature_prompt,
    novelty_collision_prompt,
    novelty_dimension_score_prompt,
    novelty_gap_prompt,
    review_prompt,
    selection_prompt,
)
from research_foundry.storage import RunStore
from research_foundry.utils import clamp_words, parse_jsonish


TOTAL_STAGES = 9


class PipelineProgress(Protocol):
    def start(self, request: ResearchRequest, total_stages: int) -> None:
        ...

    def stage_start(self, index: int, total: int, name: str, detail: str) -> None:
        ...

    def stage_tick(self, index: int, total: int, name: str) -> None:
        ...

    def stage_complete(
        self, index: int, total: int, name: str, artifact: AgentArtifact
    ) -> None:
        ...

    def stage_error(self, index: int, total: int, name: str, error: BaseException) -> None:
        ...

    def stage_skipped(
        self, index: int, total: int, name: str, reason: str, artifact: AgentArtifact
    ) -> None:
        ...

    def ideas_ready(self, ideas: list[IdeaCandidate]) -> None:
        ...

    def novelty_audit_ready(self, audits: list[IdeaNoveltyAudit]) -> None:
        ...

    def selection_ready(
        self, selection: SelectionDecision, selected_idea: IdeaCandidate
    ) -> None:
        ...

    def saved(self, report: ResearchReport, run_dir: object | None) -> None:
        ...

    def finish(self, report: ResearchReport) -> None:
        ...


class ResearchFoundry:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        settings: Settings | None = None,
        store: RunStore | None = None,
    ) -> None:
        self.gateway = gateway
        self.settings = settings or Settings()
        self.store = store or RunStore(self.settings.output_dir)

    async def run(
        self,
        request: ResearchRequest,
        *,
        save: bool = True,
        progress: PipelineProgress | None = None,
    ) -> ResearchReport:
        self._progress_call(progress, "start", request, TOTAL_STAGES)

        literature = await self._execute_stage(
            progress,
            1,
            "Literature Cartographer",
            self._literature_detail(request),
            lambda: self._run_literature(request),
        )
        gaps_artifact = await self._execute_stage(
            progress,
            2,
            "Novelty Gap Miner",
            "Extracting novelty gaps and weak spots in current work",
            lambda: self._run_gaps(request, literature),
        )
        gaps = self._parse_gaps(gaps_artifact)

        ideas_artifact = await self._execute_stage(
            progress,
            3,
            "Idea Generator",
            "Generating concrete candidate methods and paper angles",
            lambda: self._run_ideas(request, literature, gaps),
        )
        ideas = self._parse_ideas(ideas_artifact)
        self._progress_call(progress, "ideas_ready", ideas)

        novelty_audit_artifact = await self._execute_stage(
            progress,
            4,
            "Novelty Collision Auditor",
            "Searching for collisions and running independent N/P/V score auditors",
            lambda: self._run_novelty_audit(request, literature, ideas),
        )
        novelty_audits = self._parse_novelty_audits(novelty_audit_artifact, ideas)
        self._progress_call(progress, "novelty_audit_ready", novelty_audits)

        reviews_artifact = await self._execute_stage(
            progress,
            5,
            "Skeptical Review Board",
            "Running ICLR/NeurIPS-style rejection and rescue analysis",
            lambda: self._run_reviews(request, ideas, novelty_audits, literature),
        )
        reviews = self._parse_reviews(reviews_artifact)

        selection_artifact = await self._execute_stage(
            progress,
            6,
            "Best Idea Selector",
            "Choosing one research direction worth building into a paper",
            lambda: self._run_selection(
                request, literature, ideas, novelty_audits, reviews_artifact.content
            ),
        )
        selection = self._parse_selection(selection_artifact, ideas)
        self._enforce_novelty_audit_gate(request, selection, novelty_audits)
        self._enforce_ambition_gate(request, selection)
        selected_idea = self._selected_idea(ideas, selection)
        self._progress_call(progress, "selection_ready", selection, selected_idea)

        if not self.selection_gate_cleared(request, selection):
            experiment_plan, implementation_plan, final = self._rejected_selection_artifacts(
                request, selection
            )
            skip_reason = (
                f"Selected idea did not clear the {request.ambition_floor}/10 "
                f"selector gate. Selector score: {selection.score}/10."
            )
            self._skip_stage(
                progress,
                7,
                "Experiment Designer",
                skip_reason,
                experiment_plan,
            )
            self._skip_stage(
                progress,
                8,
                "Implementation Architect",
                skip_reason,
                implementation_plan,
            )
            self._skip_stage(progress, 9, "Chief Scientist", skip_reason, final)

            report = ResearchReport(
                request=request,
                literature=literature,
                novelty_gaps=gaps,
                ideas=ideas,
                novelty_audits=novelty_audits,
                reviews=reviews,
                selection=selection,
                experiment_plan=experiment_plan,
                implementation_plan=implementation_plan,
                final_recommendation=final,
                artifacts=[
                    gaps_artifact,
                    ideas_artifact,
                    novelty_audit_artifact,
                    reviews_artifact,
                    selection_artifact,
                    experiment_plan,
                    implementation_plan,
                ],
            )
            if save:
                self.store.save(report)
                self._progress_call(progress, "saved", report, self.store.last_run_dir)
            self._progress_call(progress, "finish", report)
            return report

        experiment_plan = await self._execute_stage(
            progress,
            7,
            "Experiment Designer",
            "Designing baselines, ablations, metrics, and cost accounting",
            lambda: self._run_experiments(
                request, literature, selected_idea, reviews_artifact.content, selection
            ),
        )
        implementation_plan = await self._execute_stage(
            progress,
            8,
            "Implementation Architect",
            "Writing the detailed implementation approach for the selected idea",
            lambda: self._run_implementation_plan(
                request,
                literature,
                selected_idea,
                selection,
                novelty_audits,
                reviews_artifact.content,
                experiment_plan.content,
            ),
        )
        final = await self._execute_stage(
            progress,
            9,
            "Chief Scientist",
            "Synthesizing the final research strategy and next moves",
            lambda: self._run_final(
                request,
                literature,
                gaps,
                ideas,
                novelty_audits,
                reviews_artifact.content,
                selection,
                experiment_plan.content,
                implementation_plan.content,
            ),
        )

        report = ResearchReport(
            request=request,
            literature=literature,
            novelty_gaps=gaps,
            ideas=ideas,
            novelty_audits=novelty_audits,
            reviews=reviews,
            selection=selection,
            experiment_plan=experiment_plan,
            implementation_plan=implementation_plan,
            final_recommendation=final,
            artifacts=[
                gaps_artifact,
                ideas_artifact,
                novelty_audit_artifact,
                reviews_artifact,
                selection_artifact,
                experiment_plan,
                implementation_plan,
            ],
        )
        if save:
            self.store.save(report)
            self._progress_call(progress, "saved", report, self.store.last_run_dir)
        self._progress_call(progress, "finish", report)
        return report

    async def _execute_stage(
        self,
        progress: PipelineProgress | None,
        index: int,
        name: str,
        detail: str,
        runner: Callable[[], Awaitable[AgentArtifact]],
    ) -> AgentArtifact:
        self._progress_call(progress, "stage_start", index, TOTAL_STAGES, name, detail)
        if progress is None:
            return await runner()

        task = asyncio.create_task(runner())
        while True:
            try:
                artifact = await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
            except asyncio.TimeoutError:
                self._progress_call(progress, "stage_tick", index, TOTAL_STAGES, name)
                continue
            except Exception as exc:
                self._progress_call(progress, "stage_error", index, TOTAL_STAGES, name, exc)
                raise

            self._progress_call(progress, "stage_complete", index, TOTAL_STAGES, name, artifact)
            return artifact

    def _skip_stage(
        self,
        progress: PipelineProgress | None,
        index: int,
        name: str,
        reason: str,
        artifact: AgentArtifact,
    ) -> None:
        self._progress_call(
            progress,
            "stage_skipped",
            index,
            TOTAL_STAGES,
            name,
            reason,
            artifact,
        )

    @staticmethod
    def _progress_call(
        progress: PipelineProgress | None, method_name: str, *args: object
    ) -> None:
        if progress is None:
            return
        method = getattr(progress, method_name, None)
        if method is None:
            return
        try:
            method(*args)
        except Exception:
            return

    def _literature_detail(self, request: ResearchRequest) -> str:
        if not request.use_deep_research:
            return f"Scanning literature with {self.settings.frontier_model} web search"
        model = (
            self.settings.fast_deep_research_model
            if request.fast
            else self.settings.deep_research_model
        )
        return f"Running source-grounded deep research with {model}"

    async def _run_literature(self, request: ResearchRequest) -> AgentArtifact:
        if request.use_deep_research:
            model = (
                self.settings.fast_deep_research_model
                if request.fast
                else self.settings.deep_research_model
            )
            return await self.gateway.run_text(
                agent_name="Literature Cartographer",
                prompt=literature_prompt(request),
                model=model,
                tools=[web_search_tool(deep=True)],
                background=self.settings.background_research,
                output_kind="literature",
            )

        return await self.gateway.run_text(
            agent_name="Literature Cartographer",
            prompt=literature_prompt(request),
            model=self.settings.frontier_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="literature",
        )

    async def _run_gaps(
        self, request: ResearchRequest, literature: AgentArtifact
    ) -> AgentArtifact:
        return await self.gateway.run_text(
            agent_name="Novelty Gap Miner",
            prompt=novelty_gap_prompt(request, clamp_words(literature.content, 6000)),
            model=self.settings.frontier_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="gaps",
        )

    async def _run_ideas(
        self, request: ResearchRequest, literature: AgentArtifact, gaps: list[NoveltyGap]
    ) -> AgentArtifact:
        return await self.gateway.run_text(
            agent_name="Idea Generator",
            prompt=idea_generation_prompt(request, clamp_words(literature.content, 4500), gaps),
            model=self.settings.frontier_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="ideas",
        )

    async def _run_novelty_audit(
        self, request: ResearchRequest, literature: AgentArtifact, ideas: list[IdeaCandidate]
    ) -> AgentArtifact:
        literature_text = clamp_words(literature.content, 5000)
        collision_artifact = await self.gateway.run_text(
            agent_name="Novelty Collision Auditor",
            prompt=novelty_collision_prompt(
                request,
                literature_text,
                ideas,
            ),
            model=self.settings.reviewer_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="novelty_audit",
        )
        provisional_audits = self._parse_novelty_audits(collision_artifact, ideas)
        score_artifacts = await asyncio.gather(
            *[
                self.gateway.run_text(
                    agent_name=agent_name,
                    prompt=novelty_dimension_score_prompt(
                        request,
                        literature_text,
                        ideas,
                        provisional_audits,
                        score_field=score_field,
                    ),
                    model=self.settings.reviewer_model,
                    tools=[web_search_tool(deep=False)],
                    reasoning_effort="high",
                    output_kind=output_kind,
                )
                for score_field, agent_name, output_kind in [
                    ("novelty_score", "Novelty Score Auditor", "novelty_score_audit"),
                    (
                        "paper_worth_score",
                        "Paper Worth Score Auditor",
                        "paper_worth_score_audit",
                    ),
                    (
                        "venue_upside_score",
                        "Venue Upside Score Auditor",
                        "venue_upside_score_audit",
                    ),
                ]
            ]
        )
        merged_audits = self._merge_independent_novelty_scores(
            provisional_audits,
            {
                "novelty_score": score_artifacts[0],
                "paper_worth_score": score_artifacts[1],
                "venue_upside_score": score_artifacts[2],
            },
        )
        payload = {
            "audits": [audit.model_dump(mode="json") for audit in merged_audits],
            "score_provenance": {
                "collision_auditor": collision_artifact.agent_name,
                "novelty_score": score_artifacts[0].agent_name,
                "paper_worth_score": score_artifacts[1].agent_name,
                "venue_upside_score": score_artifacts[2].agent_name,
            },
            "score_subagent_outputs": {
                "novelty_score": self._parse_dimension_scores(score_artifacts[0]),
                "paper_worth_score": self._parse_dimension_scores(score_artifacts[1]),
                "venue_upside_score": self._parse_dimension_scores(score_artifacts[2]),
            },
        }
        return AgentArtifact(
            agent_name="Novelty Collision Auditor",
            model=self._combined_model_label([collision_artifact, *score_artifacts]),
            content=json.dumps(payload, indent=2),
            sources=self._merge_sources([collision_artifact, *score_artifacts]),
            response_id=collision_artifact.response_id,
            metadata={
                "output_kind": "novelty_audit",
                "collision_response_id": collision_artifact.response_id,
                "score_response_ids": {
                    "novelty_score": score_artifacts[0].response_id,
                    "paper_worth_score": score_artifacts[1].response_id,
                    "venue_upside_score": score_artifacts[2].response_id,
                },
            },
        )

    async def _run_reviews(
        self,
        request: ResearchRequest,
        ideas: list[IdeaCandidate],
        novelty_audits: list[IdeaNoveltyAudit],
        literature: AgentArtifact,
    ) -> AgentArtifact:
        return await self.gateway.run_text(
            agent_name="Skeptical Review Board",
            prompt=review_prompt(
                request,
                ideas,
                novelty_audits,
                clamp_words(literature.content, 3500),
            ),
            model=self.settings.reviewer_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="reviews",
        )

    async def _run_selection(
        self,
        request: ResearchRequest,
        literature: AgentArtifact,
        ideas: list[IdeaCandidate],
        novelty_audits: list[IdeaNoveltyAudit],
        reviews_json: str,
    ) -> AgentArtifact:
        return await self.gateway.run_text(
            agent_name="Best Idea Selector",
            prompt=selection_prompt(
                request,
                clamp_words(literature.content, 3500),
                ideas,
                novelty_audits,
                reviews_json,
            ),
            model=self.settings.reviewer_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="selection",
        )

    async def _run_experiments(
        self,
        request: ResearchRequest,
        literature: AgentArtifact,
        selected_idea: IdeaCandidate,
        reviews_json: str,
        selection: SelectionDecision,
    ) -> AgentArtifact:
        return await self.gateway.run_text(
            agent_name="Experiment Designer",
            prompt=experiment_design_prompt(
                request,
                selected_idea,
                reviews_json,
                clamp_words(literature.content, 3000),
                selection,
            ),
            model=self.settings.frontier_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="experiments",
        )

    async def _run_implementation_plan(
        self,
        request: ResearchRequest,
        literature: AgentArtifact,
        selected_idea: IdeaCandidate,
        selection: SelectionDecision,
        novelty_audits: list[IdeaNoveltyAudit],
        reviews_json: str,
        experiment_plan: str,
    ) -> AgentArtifact:
        return await self.gateway.run_text(
            agent_name="Implementation Architect",
            prompt=implementation_doc_prompt(
                request,
                selected_idea,
                selection,
                novelty_audits,
                clamp_words(literature.content, 3500),
                reviews_json,
                experiment_plan,
            ),
            model=self.settings.frontier_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="implementation_doc",
        )

    async def _run_final(
        self,
        request: ResearchRequest,
        literature: AgentArtifact,
        gaps: list[NoveltyGap],
        ideas: list[IdeaCandidate],
        novelty_audits: list[IdeaNoveltyAudit],
        reviews_json: str,
        selection: SelectionDecision,
        experiment_plan: str,
        implementation_plan: str,
    ) -> AgentArtifact:
        return await self.gateway.run_text(
            agent_name="Chief Scientist",
            prompt=final_synthesis_prompt(
                request,
                clamp_words(literature.content, 3500),
                gaps,
                ideas,
                novelty_audits,
                reviews_json,
                selection,
                experiment_plan,
                implementation_plan,
            ),
            model=self.settings.frontier_model,
            tools=[web_search_tool(deep=False)],
            reasoning_effort="high",
            output_kind="final",
        )

    def _parse_gaps(self, artifact: AgentArtifact) -> list[NoveltyGap]:
        try:
            data = parse_jsonish(artifact.content)
            items = data.get("gaps", data) if isinstance(data, dict) else data
            return [NoveltyGap.model_validate(item) for item in items]
        except Exception:
            return [
                NoveltyGap(
                    gap=clamp_words(artifact.content, 80),
                    why_now="The model returned unstructured gap analysis.",
                    nearest_prior_work=[],
                    evidence_needed=["Manually inspect the raw gap artifact."],
                    risk="high",
                )
            ]

    def _parse_ideas(self, artifact: AgentArtifact) -> list[IdeaCandidate]:
        try:
            data = parse_jsonish(artifact.content)
            items = data.get("ideas", data) if isinstance(data, dict) else data
            return [IdeaCandidate.model_validate(item) for item in items]
        except Exception:
            return [
                IdeaCandidate(
                    title="Unparsed Idea Portfolio",
                    thesis=clamp_words(artifact.content, 80),
                    core_mechanism="Inspect raw artifact.",
                    novelty_claim="Unknown until parsed.",
                    why_reviewers_care="Unknown until parsed.",
                    expected_evidence=["Manually inspect the raw idea artifact."],
                    first_experiments=[],
                    key_risks=["Model output was not valid JSON."],
                )
            ]

    def _parse_novelty_audits(
        self, artifact: AgentArtifact, ideas: list[IdeaCandidate]
    ) -> list[IdeaNoveltyAudit]:
        try:
            data = parse_jsonish(artifact.content)
            items = data.get("audits", data) if isinstance(data, dict) else data
            return [IdeaNoveltyAudit.model_validate(item) for item in items]
        except Exception:
            return [
                IdeaNoveltyAudit(
                    idea_title=idea.title,
                    novelty_score=1,
                    paper_worth_score=1,
                    venue_upside_score=1,
                    main_track_verdict="fail",
                    workshop_risk=True,
                    main_track_blockers=[
                        "Novelty auditor output was invalid, so this idea is not cleared for a main-track paper."
                    ],
                    required_reframing=[
                        "Rerun the novelty collision audit with web search and inspect closest prior work before selection."
                    ],
                    direct_comparisons_required=[],
                )
                for idea in ideas
            ]

    @staticmethod
    def _parse_dimension_scores(artifact: AgentArtifact) -> dict[str, dict[str, object]]:
        try:
            data = parse_jsonish(artifact.content)
            items = data.get("scores", data) if isinstance(data, dict) else data
            parsed: dict[str, dict[str, object]] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("idea_title", "")).strip()
                if not title:
                    continue
                score = int(item.get("score"))
                if not 1 <= score <= 10:
                    continue
                parsed[title] = {
                    "score": score,
                    "rationale": item.get("rationale", ""),
                    "risks": item.get("risks", []),
                }
            return parsed
        except Exception:
            return {}

    def _merge_independent_novelty_scores(
        self,
        audits: list[IdeaNoveltyAudit],
        score_artifacts: dict[str, AgentArtifact],
    ) -> list[IdeaNoveltyAudit]:
        score_maps = {
            score_field: self._parse_dimension_scores(artifact)
            for score_field, artifact in score_artifacts.items()
        }
        merged: list[IdeaNoveltyAudit] = []
        for audit in audits:
            data = audit.model_dump(mode="json")
            for score_field, scores_by_title in score_maps.items():
                score_info = scores_by_title.get(audit.idea_title)
                if score_info is None:
                    continue
                data[score_field] = score_info["score"]
            merged.append(IdeaNoveltyAudit.model_validate(data))
        return merged

    @staticmethod
    def _combined_model_label(artifacts: list[AgentArtifact]) -> str:
        labels: list[str] = []
        for artifact in artifacts:
            if artifact.model not in labels:
                labels.append(artifact.model)
        return " + ".join(labels)

    @staticmethod
    def _merge_sources(artifacts: list[AgentArtifact]) -> list[SourceRef]:
        sources: list[SourceRef] = []
        seen: set[str] = set()
        for artifact in artifacts:
            for source in artifact.sources:
                if source.url in seen:
                    continue
                seen.add(source.url)
                sources.append(source)
        return sources

    def _parse_reviews(self, artifact: AgentArtifact) -> list[ReviewScore]:
        try:
            data = parse_jsonish(artifact.content)
            items = data.get("reviews", data) if isinstance(data, dict) else data
            return [ReviewScore.model_validate(item) for item in items]
        except Exception:
            return []

    def _parse_selection(
        self, artifact: AgentArtifact, ideas: list[IdeaCandidate]
    ) -> SelectionDecision:
        try:
            data = parse_jsonish(artifact.content)
            selection = SelectionDecision.model_validate(data)
            if any(idea.title == selection.selected_title for idea in ideas):
                return selection
        except Exception:
            pass

        fallback = ideas[0] if ideas else None
        title = fallback.title if fallback else "No parsed idea"
        return SelectionDecision(
            selected_title=title,
            rationale="Fallback selection because the selector output could not be parsed.",
            iclr_neurips_case="Requires manual inspection before treating this as paper-worthy.",
            decisive_strengths=[],
            decisive_risks=["Selector output was missing or invalid."],
            required_next_steps=["Inspect raw selector artifact and rerun selection."],
            score=1,
        )

    @staticmethod
    def _selected_idea(
        ideas: list[IdeaCandidate], selection: SelectionDecision
    ) -> IdeaCandidate:
        for idea in ideas:
            if idea.title == selection.selected_title:
                return idea
        if ideas:
            return ideas[0]
        return IdeaCandidate(
            title=selection.selected_title,
            thesis="No parsed idea was available.",
            core_mechanism="Unknown.",
            novelty_claim="Unknown.",
            why_reviewers_care="Unknown.",
            key_risks=["No idea parsed."],
        )

    @staticmethod
    def selection_gate_cleared(
        request: ResearchRequest, selection: SelectionDecision
    ) -> bool:
        floor = request.ambition_floor
        required_scores = [
            selection.score,
            selection.research_worth_score,
            selection.paper_worth_score,
            selection.venue_upside_score,
            selection.fixed_pool_only_score,
        ]
        if any(value is None or value < floor for value in required_scores):
            return False
        return all(
            check.score >= floor for check in selection.main_conference_checklist
        )

    @staticmethod
    def _rejected_selection_artifacts(
        request: ResearchRequest, selection: SelectionDecision
    ) -> tuple[AgentArtifact, AgentArtifact, AgentArtifact]:
        metadata = {
            "output_kind": "selector_gate_skip",
            "skipped_due_to_selector_gate": True,
            "selector_gate_cleared": False,
            "selector_score": selection.score,
            "ambition_floor": request.ambition_floor,
        }
        reasons = "\n".join(f"- {risk}" for risk in selection.decisive_risks) or "-"
        next_steps = (
            "\n".join(f"- {step}" for step in selection.required_next_steps) or "-"
        )
        content = (
            "# Selector Gate Not Cleared\n\n"
            f"Selected idea: {selection.selected_title}\n\n"
            "Gate status: not cleared\n\n"
            f"Selector score: {selection.score}/10\n\n"
            f"Required gate: selector score, research worth, paper worth, venue upside, "
            f"fixed-pool-only upside, and every checklist row must clear "
            f"{request.ambition_floor}/10.\n\n"
            "This batch was rejected by the selector gate. Downstream experiment "
            "design, implementation planning, and final paper synthesis were skipped "
            "so the orchestrator can generate a fresh idea batch instead.\n\n"
            "## Blocking Reasons\n\n"
            f"{reasons}\n\n"
            "## Required Repair Moves\n\n"
            f"{next_steps}\n"
        )
        experiment_plan = AgentArtifact(
            agent_name="Experiment Designer",
            model="pipeline-gate",
            content=content,
            metadata=metadata.copy(),
        )
        implementation_plan = AgentArtifact(
            agent_name="Implementation Architect",
            model="pipeline-gate",
            content=content,
            metadata=metadata.copy(),
        )
        final = AgentArtifact(
            agent_name="Chief Scientist",
            model="pipeline-gate",
            content=content,
            metadata=metadata.copy(),
        )
        return experiment_plan, implementation_plan, final

    @staticmethod
    def _enforce_ambition_gate(
        request: ResearchRequest, selection: SelectionDecision
    ) -> None:
        scores = {
            "research worth": selection.research_worth_score,
            "paper worth": selection.paper_worth_score,
            "venue upside": selection.venue_upside_score,
            "fixed-pool only": selection.fixed_pool_only_score,
        }
        scored_values = [value for value in scores.values() if value is not None]
        if not scored_values:
            return

        floor = request.ambition_floor
        missing = [
            f"{label}={value}/10" if value is not None else f"{label}=missing"
            for label, value in scores.items()
            if value is None or value < floor
        ]
        missing.extend(
            f"checklist: {check.name}={check.score}/10"
            for check in selection.main_conference_checklist
            if check.score < floor
        )
        if not missing:
            return

        selection.score = min(selection.score, max(1, floor - 1))
        gate_note = (
            "Does not clear the ambition floor for: "
            + ", ".join(missing)
            + f". Required floor is {floor}/10."
        )
        if gate_note not in selection.decisive_risks:
            selection.decisive_risks.insert(0, gate_note)
        next_step = (
            "Regenerate or revise the idea until research worth, paper worth, venue upside, "
            "and fixed-pool-only upside all clear the ambition floor."
        )
        if next_step not in selection.required_next_steps:
            selection.required_next_steps.insert(0, next_step)

    @staticmethod
    def _enforce_novelty_audit_gate(
        request: ResearchRequest,
        selection: SelectionDecision,
        novelty_audits: list[IdeaNoveltyAudit],
    ) -> None:
        matching_audit = next(
            (audit for audit in novelty_audits if audit.idea_title == selection.selected_title),
            None,
        )
        if matching_audit is None:
            return

        floor = request.ambition_floor
        reasons: list[str] = []
        cap: int | None = None

        if matching_audit.main_track_verdict == "fail":
            reasons.append("novelty audit verdict is fail")
            cap = 5
        elif matching_audit.main_track_verdict == "borderline":
            reasons.append("novelty audit verdict is borderline")
            cap = 7

        fatal_collisions = [
            collision.prior_work
            for collision in matching_audit.closest_prior_work
            if collision.severity == "fatal"
        ]
        if fatal_collisions:
            reasons.append(
                "fatal novelty collisions: " + ", ".join(fatal_collisions[:5])
            )
            cap = min(cap or 5, 5)

        high_collisions = [
            collision.prior_work
            for collision in matching_audit.closest_prior_work
            if collision.severity == "high"
        ]
        if high_collisions and matching_audit.main_track_verdict != "pass":
            reasons.append("high novelty collisions: " + ", ".join(high_collisions[:5]))
            cap = min(cap or 7, 7)

        score_failures = [
            label
            for label, value in [
                ("novelty", matching_audit.novelty_score),
                ("paper worth", matching_audit.paper_worth_score),
                ("venue upside", matching_audit.venue_upside_score),
            ]
            if value < floor
        ]
        if score_failures:
            reasons.append("audit scores below floor: " + ", ".join(score_failures))
            cap = min(cap or floor - 1, floor - 1)

        if not reasons:
            if high_collisions:
                gate_note = (
                    "High novelty collisions remain but the audit marked the selected idea "
                    "as pass with scores above the ambition floor: "
                    + ", ".join(high_collisions[:5])
                    + ". Treat these as mandatory positioning and baseline risks, not an "
                    "automatic selector-score cap."
                )
                if gate_note not in selection.decisive_risks:
                    selection.decisive_risks.insert(0, gate_note)
                for blocker in reversed(matching_audit.main_track_blockers[:4]):
                    note = "Resolve novelty blocker: " + blocker
                    if note not in selection.required_next_steps:
                        selection.required_next_steps.insert(0, note)
                for comparison in matching_audit.direct_comparisons_required[:4]:
                    note = "Add direct comparison against: " + comparison
                    if note not in selection.required_next_steps:
                        selection.required_next_steps.append(note)
            return

        selection.score = min(selection.score, max(1, cap or floor - 1))
        gate_note = (
            "Novelty collision gate not cleared: "
            + "; ".join(reasons)
            + f". Required floor is {floor}/10."
        )
        if gate_note not in selection.decisive_risks:
            selection.decisive_risks.insert(0, gate_note)

        for blocker in reversed(matching_audit.main_track_blockers[:4]):
            note = "Resolve novelty blocker: " + blocker
            if note not in selection.required_next_steps:
                selection.required_next_steps.insert(0, note)

        for comparison in matching_audit.direct_comparisons_required[:4]:
            note = "Add direct comparison against: " + comparison
            if note not in selection.required_next_steps:
                selection.required_next_steps.append(note)

    @staticmethod
    def report_summary(report: ResearchReport) -> str:
        top = report.top_idea.title if report.top_idea else "No idea parsed"
        selected_audit = next(
            (
                audit
                for audit in report.novelty_audits
                if audit.idea_title == report.selection.selected_title
            ),
            None,
        )
        novelty_audit_cleared = (
            selected_audit is None
            or (
                selected_audit.main_track_verdict == "pass"
                and selected_audit.novelty_score >= report.request.ambition_floor
                and selected_audit.paper_worth_score >= report.request.ambition_floor
                and selected_audit.venue_upside_score >= report.request.ambition_floor
            )
        )
        return json.dumps(
            {
                "top_idea": top,
                "selected_idea": report.selection.selected_title,
                "ambition_floor": report.request.ambition_floor,
                "selector_score": report.selection.score,
                "selection_scores": {
                    "research_worth": report.selection.research_worth_score,
                    "paper_worth": report.selection.paper_worth_score,
                    "venue_upside": report.selection.venue_upside_score,
                    "fixed_pool_only": report.selection.fixed_pool_only_score,
                },
                "selector_gate_cleared": ResearchFoundry.selection_gate_cleared(
                    report.request, report.selection
                )
                and novelty_audit_cleared,
                "ambition_floor_cleared": all(
                    value is not None and value >= report.request.ambition_floor
                    for value in [
                        report.selection.score,
                        report.selection.research_worth_score,
                        report.selection.paper_worth_score,
                        report.selection.venue_upside_score,
                        report.selection.fixed_pool_only_score,
                    ]
                )
                and all(
                    check.score >= report.request.ambition_floor
                    for check in report.selection.main_conference_checklist
                )
                and novelty_audit_cleared,
                "main_conference_checks": [
                    {
                        "name": check.name,
                        "score": check.score,
                        "passed": check.score >= check.threshold,
                    }
                    for check in report.selection.main_conference_checklist
                ],
                "implementation_docx": report.implementation_docx_path,
                "ideas": [idea.title for idea in report.ideas],
                "novelty_audit": [
                    {
                        "idea": audit.idea_title,
                        "verdict": audit.main_track_verdict,
                        "novelty": audit.novelty_score,
                        "paper": audit.paper_worth_score,
                        "venue": audit.venue_upside_score,
                    }
                    for audit in report.novelty_audits
                ],
                "review_count": len(report.reviews),
                "literature_sources": len(report.literature.sources),
            },
            indent=2,
        )
