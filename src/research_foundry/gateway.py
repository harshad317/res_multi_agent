from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

from research_foundry.config import Settings
from research_foundry.models import AgentArtifact, SourceRef


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}
FORCE_HIGH_REASONING_MODELS = {"gpt-5.5", "gpt-5.5-pro"}


class ModelGateway(Protocol):
    async def run_text(
        self,
        *,
        agent_name: str,
        prompt: str,
        model: str,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        background: bool = False,
        reasoning_effort: str | None = None,
        output_kind: str | None = None,
    ) -> AgentArtifact:
        ...


def response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return dict(response)


def extract_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct)

    data = response_to_dict(response)
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def extract_sources(response: Any) -> list[SourceRef]:
    data = response_to_dict(response)
    seen: set[str] = set()
    sources: list[SourceRef] = []

    def add(url: str | None, title: str | None = None, note: str | None = None) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        sources.append(SourceRef(title=title, url=url, note=note))

    for item in data.get("output", []):
        for source in item.get("sources", []) or []:
            add(source.get("url"), source.get("title"), "web_search source")
        for content in item.get("content", []) or []:
            for annotation in content.get("annotations", []) or []:
                if annotation.get("type") == "url_citation":
                    add(annotation.get("url"), annotation.get("title"), "inline citation")

    return sources


def web_search_tool(*, deep: bool = False) -> dict[str, Any]:
    # Keep the flag for stage semantics, but deep research models reject higher context sizes.
    _ = deep
    return {
        "type": "web_search",
        "search_context_size": "medium",
    }


def requires_high_reasoning(model: str) -> bool:
    return model.strip().lower() in FORCE_HIGH_REASONING_MODELS


def reasoning_effort_for_model(model: str, requested_effort: str | None = None) -> str | None:
    if requires_high_reasoning(model):
        return "high"
    return requested_effort


class OpenAIResponsesGateway:
    """Thin wrapper over the Responses API with polling and citation extraction."""

    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required for live runs. Use --dry-run to test locally."
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def run_text(
        self,
        *,
        agent_name: str,
        prompt: str,
        model: str,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        background: bool = False,
        reasoning_effort: str | None = None,
        output_kind: str | None = None,
    ) -> AgentArtifact:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "metadata": {
                "agent_name": agent_name,
                "output_kind": output_kind or "text",
                "app": "research_foundry",
            },
        }
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = tools
        if background:
            kwargs["background"] = True
        effective_reasoning_effort = reasoning_effort_for_model(model, reasoning_effort)
        if effective_reasoning_effort:
            kwargs["reasoning"] = {"effort": effective_reasoning_effort}

        response = await self.client.responses.create(**kwargs)
        response = await self._poll_if_needed(response)

        data = response_to_dict(response)
        status = data.get("status")
        if status and status != "completed":
            raise RuntimeError(f"Response ended with status={status}: {data.get('error')}")

        return AgentArtifact(
            agent_name=agent_name,
            model=model,
            content=extract_response_text(response),
            sources=extract_sources(response),
            response_id=data.get("id"),
            metadata={
                "status": status,
                "output_kind": output_kind,
                "usage": data.get("usage"),
                "background": data.get("background", background),
            },
        )

    async def _poll_if_needed(self, response: Any) -> Any:
        data = response_to_dict(response)
        if data.get("status") in TERMINAL_STATUSES:
            return response
        response_id = data.get("id")
        if not response_id:
            return response

        deadline = time.monotonic() + self.settings.max_wait_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(self.settings.poll_seconds)
            response = await self.client.responses.retrieve(response_id)
            data = response_to_dict(response)
            if data.get("status") in TERMINAL_STATUSES:
                return response

        raise TimeoutError(
            f"Timed out waiting for background response {response_id} after "
            f"{self.settings.max_wait_seconds}s."
        )


class DryRunGateway:
    """Deterministic gateway for local development and tests."""

    async def run_text(
        self,
        *,
        agent_name: str,
        prompt: str,
        model: str,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        background: bool = False,
        reasoning_effort: str | None = None,
        output_kind: str | None = None,
    ) -> AgentArtifact:
        _ = (prompt, instructions, tools, background, reasoning_effort)
        content = self._content_for(output_kind, agent_name)
        return AgentArtifact(
            agent_name=agent_name,
            model=f"dry-run:{model}",
            content=content,
            sources=[],
            response_id=None,
            metadata={"dry_run": True, "output_kind": output_kind},
        )

    def _content_for(self, output_kind: str | None, agent_name: str) -> str:
        if output_kind == "gaps":
            return json.dumps(
                {
                    "gaps": [
                        {
                            "gap": "Current agent benchmarks reward task completion but rarely measure scientific taste, falsifiability, or citation-grounded novelty.",
                            "why_now": "Deep research models can now gather source-grounded context, making novelty-aware evaluation more practical.",
                            "nearest_prior_work": [
                                "Agent evaluation benchmarks",
                                "LLM-as-reviewer studies",
                                "Automated scientific discovery systems",
                            ],
                            "evidence_needed": [
                                "Correlation with expert novelty judgments",
                                "Ablation showing citation graph features matter",
                            ],
                            "risk": "medium",
                        }
                    ]
                },
                indent=2,
            )
        if output_kind == "ideas":
            return json.dumps(
                {
                    "ideas": [
                        {
                            "title": "Novelty Stress Tests for Scientific Agent Systems",
                            "thesis": "Research agents should be evaluated by whether their ideas survive source-grounded novelty attacks, not only by whether they produce plausible proposals.",
                            "core_mechanism": "A benchmark that pairs generated paper ideas with adversarial retrieval, must-cite detection, and reviewer-style falsification rubrics.",
                            "novelty_claim": "Unlike generic agent benchmarks, the task targets research taste and novelty under citation pressure.",
                            "why_reviewers_care": "It gives the community a concrete way to measure whether AI systems help produce genuinely useful research directions.",
                            "nearest_prior_work": [
                                "Agent evaluation benchmarks",
                                "LLM-as-reviewer studies",
                            ],
                            "decisive_differentiator": "Turns novelty into an adversarial, source-grounded survival test rather than a generic review score.",
                            "innovation_score": 8,
                            "paper_worth_score": 8,
                            "venue_upside_score": 8,
                            "fixed_pool_only_score": 8,
                            "expected_evidence": [
                                "Agreement with expert reviewer novelty labels",
                                "Failure-mode taxonomy showing why plausible ideas are rejected",
                            ],
                            "first_experiments": [
                                "Collect 100 accepted and 100 rejected ML paper abstracts with review metadata.",
                                "Compare generic LLM review, retrieval-only review, and novelty stress-test agents.",
                            ],
                            "key_risks": [
                                "Expert-label cost",
                                "Reviewer subjectivity",
                                "Benchmark contamination",
                            ],
                            "citations": [],
                        },
                        {
                            "title": "Citation-Causal Idea Search for ML Paper Generation",
                            "thesis": "Idea generation improves when agents optimize against a causal graph of prior claims rather than a flat list of related papers.",
                            "core_mechanism": "Build a claim graph from literature, identify unsupported edges or missing interventions, then propose experiments that break or extend those edges.",
                            "novelty_claim": "The contribution is a source-grounded search procedure over scientific claims, not another brainstorming prompt.",
                            "why_reviewers_care": "It converts literature review into executable hypothesis search.",
                            "nearest_prior_work": [
                                "Retrieval-augmented ideation systems",
                                "Claim extraction for scientific literature",
                            ],
                            "decisive_differentiator": "Searches over gaps in a structured claim graph instead of sampling ideas from a flat literature summary.",
                            "innovation_score": 8,
                            "paper_worth_score": 7,
                            "venue_upside_score": 7,
                            "fixed_pool_only_score": 5,
                            "expected_evidence": [
                                "Higher expert-rated novelty than retrieval-augmented baselines",
                                "Better must-cite coverage",
                            ],
                            "first_experiments": [
                                "Run on agentic AI, robustness, and evaluation subfields.",
                                "Measure novelty, feasibility, and baseline coverage with blinded reviewers.",
                            ],
                            "key_risks": [
                                "Claim extraction errors",
                                "Hard to prove causality in literature graphs",
                            ],
                            "citations": [],
                        },
                    ]
                },
                indent=2,
            )
        if output_kind == "novelty_audit":
            return json.dumps(
                {
                    "audits": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "closest_prior_work": [
                                {
                                    "prior_work": "LLM-as-reviewer studies",
                                    "url": "https://example.com/llm-reviewer",
                                    "overlap": "Prior work already uses LLMs to review scientific ideas, but not as a source-grounded novelty stress-test benchmark.",
                                    "severity": "medium",
                                    "what_to_cede": "Do not claim first use of LLMs for paper review.",
                                    "required_differentiator": "Show adversarial retrieval and novelty-survival scoring predict expert novelty objections better than reviewer baselines.",
                                }
                            ],
                            "novelty_score": 8,
                            "paper_worth_score": 8,
                            "venue_upside_score": 8,
                            "main_track_verdict": "pass",
                            "workshop_risk": False,
                            "main_track_blockers": [],
                            "required_reframing": [
                                "Frame as a novelty-survival benchmark and method, not generic LLM review."
                            ],
                            "direct_comparisons_required": [
                                "Generic LLM reviewer",
                                "Retrieval-augmented reviewer",
                            ],
                        },
                        {
                            "idea_title": "Citation-Causal Idea Search for ML Paper Generation",
                            "closest_prior_work": [
                                {
                                    "prior_work": "Retrieval-augmented scientific ideation systems",
                                    "url": "https://example.com/retrieval-ideation",
                                    "overlap": "Claim-graph retrieval is close to existing literature-grounded ideation.",
                                    "severity": "high",
                                    "what_to_cede": "Do not claim source-grounded ideation itself is new.",
                                    "required_differentiator": "Validate causal-graph interventions rather than ordinary citation retrieval.",
                                }
                            ],
                            "novelty_score": 6,
                            "paper_worth_score": 6,
                            "venue_upside_score": 6,
                            "main_track_verdict": "fail",
                            "workshop_risk": True,
                            "main_track_blockers": [
                                "Likely to be seen as retrieval-augmented ideation with stronger prompting."
                            ],
                            "required_reframing": [
                                "Narrow to a validated claim-intervention benchmark."
                            ],
                            "direct_comparisons_required": [
                                "Retrieval-augmented ideation baseline",
                                "Claim extraction baseline",
                            ],
                        },
                    ],
                    "batch_verdict": "pass",
                    "batch_rationale": "One idea clears the main-track novelty bar; one should be rejected or reframed.",
                },
                indent=2,
            )
        if output_kind == "novelty_score_audit":
            return json.dumps(
                {
                    "scores": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "score": 8,
                            "rationale": "The novelty-survival framing is distinct from generic LLM review.",
                            "risks": ["Could still be seen as an evaluation benchmark."],
                        },
                        {
                            "idea_title": "Citation-Causal Idea Search for ML Paper Generation",
                            "score": 6,
                            "rationale": "The claim-graph search idea remains close to retrieval-augmented ideation.",
                            "risks": ["Causal framing may not survive prior-work pressure."],
                        },
                    ]
                },
                indent=2,
            )
        if output_kind == "paper_worth_score_audit":
            return json.dumps(
                {
                    "scores": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "score": 8,
                            "rationale": "A positive expert-alignment MVP would justify a serious paper.",
                            "risks": ["Expert annotation reliability must be demonstrated."],
                        },
                        {
                            "idea_title": "Citation-Causal Idea Search for ML Paper Generation",
                            "score": 6,
                            "rationale": "The MVP may not distinguish the method from retrieval baselines.",
                            "risks": ["Positive results could look like stronger retrieval."],
                        },
                    ]
                },
                indent=2,
            )
        if output_kind == "venue_upside_score_audit":
            return json.dumps(
                {
                    "scores": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "score": 8,
                            "rationale": "The benchmark could matter to the broader agent-evaluation community.",
                            "risks": ["Venue upside depends on expert validation breadth."],
                        },
                        {
                            "idea_title": "Citation-Causal Idea Search for ML Paper Generation",
                            "score": 6,
                            "rationale": "The contribution ceiling is limited by close ideation-system prior work.",
                            "risks": ["Reviewers may classify it as tooling."],
                        },
                    ]
                },
                indent=2,
            )
        if output_kind == "reviews":
            return json.dumps(
                {
                    "reviews": [
                        {
                            "idea_title": "Novelty Stress Tests for Scientific Agent Systems",
                            "reviewer": "ICLR Reviewer 2",
                            "novelty": 8,
                            "significance": 8,
                            "correctness": 6,
                            "feasibility": 7,
                            "clarity": 8,
                            "research_worth_score": 8,
                            "paper_worth_score": 8,
                            "venue_upside_score": 8,
                            "fixed_pool_only_score": 8,
                            "recommendation": "accept",
                            "fatal_flaws": [
                                "Needs evidence that novelty labels are stable across expert reviewers."
                            ],
                            "required_experiments": [
                                "Inter-annotator agreement and ablation of retrieval depth."
                            ],
                            "rescue_moves": [
                                "Pre-register rubric and release a reproducible benchmark subset."
                            ],
                        },
                        {
                            "idea_title": "Citation-Causal Idea Search for ML Paper Generation",
                            "reviewer": "NeurIPS Area Chair",
                            "novelty": 7,
                            "significance": 8,
                            "correctness": 5,
                            "feasibility": 6,
                            "clarity": 6,
                            "research_worth_score": 7,
                            "paper_worth_score": 7,
                            "venue_upside_score": 7,
                            "fixed_pool_only_score": 5,
                            "recommendation": "borderline",
                            "fatal_flaws": [
                                "The causal-graph claim may be too strong unless carefully scoped."
                            ],
                            "required_experiments": [
                                "Compare against strong retrieval-augmented ideation baselines."
                            ],
                            "rescue_moves": [
                                "Frame as claim-graph search instead of causal discovery unless validated."
                            ],
                        },
                    ]
                },
                indent=2,
            )
        if output_kind == "selection":
            return json.dumps(
                {
                    "selected_title": "Novelty Stress Tests for Scientific Agent Systems",
                    "rationale": "It is the strongest paper bet because it has a concrete artifact, clear baselines, measurable novelty claims, and direct relevance to the evaluation of scientific agents.",
                    "iclr_neurips_case": "A source-grounded benchmark for novelty stress testing would be timely, empirically defensible, and useful to the broader agent and evaluation community.",
                    "research_worth_score": 8,
                    "paper_worth_score": 8,
                    "venue_upside_score": 8,
                    "fixed_pool_only_score": 8,
                    "breakthrough_condition": "Expert-calibrated novelty stress tests must predict reviewer novelty objections better than retrieval-augmented reviewer baselines.",
                    "decisive_strengths": [
                        "Clear evaluation contribution",
                        "Direct path to expert validation",
                        "Strong fit with current concern about AI-generated research quality",
                    ],
                    "decisive_risks": [
                        "Expert labels may be expensive or noisy",
                        "Reviewers may ask whether this is benchmark engineering rather than a method",
                    ],
                    "required_next_steps": [
                        "Run a small expert-label pilot",
                        "Compare against retrieval-augmented reviewer baselines",
                        "Define a reproducible novelty rubric",
                    ],
                    "score": 8,
                },
                indent=2,
            )
        if output_kind == "experiments":
            return """# Experiment Plan

## Baseline Matrix
| Question | Baseline | Metric | Reviewer Objection Answered |
| --- | --- | --- | --- |
| Does novelty stress testing help? | Generic LLM reviewer | Expert agreement | Shows value over prompting alone |
| Does retrieval matter? | No-retrieval reviewer | Must-cite recall | Shows source grounding matters |

## First Minimum Viable Experiment
Create a 40-paper pilot corpus, label novelty judgments with two expert annotators, and compare three reviewer agents: generic, retrieval-augmented, and novelty-stress-test.

## Critical Ablations
- Remove citation graph features.
- Remove adversarial nearest-prior-work retrieval.
- Swap expert rubric for generic acceptance prediction.

## Reproducibility
Release prompts, source lists, annotation rubric, model settings, and bootstrap confidence intervals.
"""
        if output_kind == "implementation_doc":
            return """# Implementation Plan: Novelty Stress Tests for Scientific Agent Systems

## Paper Thesis
Current research agents can produce plausible ideas, but the field lacks a rigorous way to test whether those ideas survive citation-grounded novelty attacks. The paper introduces a benchmark and method for stress-testing generated research ideas against nearest prior work, must-cite coverage, and reviewer-style falsification.

## Method Design
The system has four stages:
1. Generate a candidate research idea.
2. Retrieve nearest prior work and must-cite papers.
3. Construct adversarial novelty attacks: already-done, trivial-extension, missing-baseline, and weak-evidence attacks.
4. Score whether the idea survives and identify the minimum experiments needed to make it paper-worthy.

## Algorithm Sketch
Input: research idea, target venue, field constraints, retrieval budget.
Output: novelty survival score, fatal flaws, rescue plan.

For each idea, retrieve K nearest papers, extract claims, compare mechanisms, score overlap, then ask reviewer agents to write rejection arguments grounded in the retrieved evidence. Aggregate survival scores with expert-calibrated weights.

## Code Structure
- `retrieval/`: paper search, metadata normalization, PDF/abstract cache.
- `claims/`: claim extraction and claim graph construction.
- `attacks/`: novelty attack generators.
- `scoring/`: cost-normalized and expert-alignment metrics.
- `experiments/`: benchmark runners and ablation scripts.
- `paper/`: tables, plots, and report generation.

## Baselines
Compare against generic LLM reviewers, retrieval-augmented reviewers, simple citation overlap, and acceptance-prediction prompts.

## Six-Week Plan
Week 1: build corpus and rubric.
Week 2: implement retrieval and claim extraction.
Week 3: implement novelty attacks and scoring.
Week 4: run baselines and ablations.
Week 5: expert evaluation and statistical tests.
Week 6: write paper and release artifacts.
"""
        if output_kind == "final":
            return """# Final Research Strategy

## Ranking
1. Novelty Stress Tests for Scientific Agent Systems
2. Citation-Causal Idea Search for ML Paper Generation

## Best Bet
The strongest first submission target is **Novelty Stress Tests for Scientific Agent Systems** because it is concrete, benchmarkable, and directly aligned with the community's need for better agent evaluation.

## 14-Day Plan
- Days 1-3: define novelty and falsifiability rubric.
- Days 4-7: assemble seed corpus and must-cite graph.
- Days 8-10: implement baseline reviewers.
- Days 11-14: run pilot annotation and measure agreement.

## Kill Criteria
Stop or pivot if expert novelty labels show low agreement even after rubric calibration, or if retrieval-augmented baselines already match the proposed stress-test system.

## LLM Usage Disclosure
LLMs were used for ideation support, literature triage, and draft planning. Human authors selected the final research question, verified sources, designed experiments, and wrote the final submission.
"""
        return f"""# {agent_name} Dry Run Literature Map

This dry-run artifact stands in for an `o3-deep-research` literature scan. In a live run, this step should gather current papers, must-cite baselines, OpenReview discussions, benchmark pages, and source-grounded danger zones before idea generation.
"""
