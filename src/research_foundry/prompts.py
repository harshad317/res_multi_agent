from __future__ import annotations

import json

from research_foundry.models import (
    IdeaCandidate,
    IdeaNoveltyAudit,
    NoveltyGap,
    ResearchRequest,
    SelectionDecision,
)


MAIN_CONFERENCE_CHECKS = [
    "Originality against nearest prior work",
    "Significance to the ICLR/NeurIPS community",
    "Technical depth beyond an engineering wrapper",
    "Mechanistic insight or theory of why it works",
    "Strong baseline and SOTA comparison plan",
    "Evaluation breadth across tasks, seeds, and regimes",
    "Statistical reliability and uncertainty reporting",
    "Ablations and negative controls that isolate the contribution",
    "Reproducibility and artifact readiness",
    "Reviewer objection survivability",
    "Main-paper narrative clarity",
    "MVP evidence strong enough to justify scaling",
]


def main_conference_checklist_instruction(threshold: int) -> str:
    checks = "\n".join(f"- {item}" for item in MAIN_CONFERENCE_CHECKS)
    return f"""Main-conference checklist:
Score every serious idea on these dimensions using 1-10 scores, with threshold {threshold}/10:
{checks}

For each dimension, include the evidence needed and the failure mode that would make reviewers reject the paper.
"""


def request_context(request: ResearchRequest) -> str:
    constraints = "\n".join(f"- {item}" for item in request.constraints) or "- None supplied"
    venues = ", ".join(request.target_venues)
    return f"""Field:
{request.field}

Objective:
{request.objective}

Target venues:
{venues}

Risk tolerance:
{request.risk_tolerance}

Constraints:
{constraints}

Desired number of final ideas:
{request.idea_count}

Ambition floor:
Every final idea should be capable of scoring at least {request.ambition_floor}/10 on novelty,
significance, paper worth, and ICLR/NeurIPS upside if the key experiment succeeds. If the
literature does not support that level of upside, say so explicitly instead of inflating the idea.
"""


def literature_prompt(request: ResearchRequest) -> str:
    return f"""You are the Literature Cartographer for a serious ML research lab.

Your job is to run deep research before anyone proposes ideas.

{request_context(request)}

Produce a citation-rich technical map. Cover:
1. The nearest must-cite papers, benchmarks, systems, and negative results.
2. What has already been tried and why it was insufficient.
3. Strong baselines a reviewer would expect.
4. Underexplored assumptions, brittle evaluation norms, and missing ablations.
5. What would make a contribution feel original at ICLR or NeurIPS.
6. A short list of "danger zones" where a proposed paper would look incremental.

Rules:
- Do not invent papers, authors, venues, benchmarks, or URLs.
- Prefer primary sources: conference proceedings, arXiv, OpenReview, official benchmark pages, and project repositories.
- If evidence is uncertain, mark it as uncertain.
- Make the output useful for downstream novelty mining, not merely a survey.
"""


def novelty_gap_prompt(request: ResearchRequest, literature: str) -> str:
    schema = {
        "gaps": [
            {
                "gap": "Precise missing research opportunity.",
                "why_now": "Why the timing is unusually good.",
                "nearest_prior_work": ["Paper or system names, not fabricated."],
                "evidence_needed": ["Evidence that would convince a skeptical reviewer."],
                "risk": "low|medium|high",
                "breakthrough_potential": 1,
                "why_not_incremental": "What makes this more than an engineering extension.",
            }
        ]
    }
    return f"""You are the Novelty Gap Miner.

{request_context(request)}

Literature map:
{literature}

Find non-obvious, review-survivable gaps. Prefer gaps that are:
- technically crisp,
- empirically testable,
- meaningfully different from nearest prior work,
- feasible for a small research team,
- likely to produce a clear ICLR/NeurIPS contribution.

Reject gaps that are merely wrappers, schedulers, benchmark variants, or prompt-engineering
heuristics unless they expose a new mechanism, measurement principle, theory, or empirical
phenomenon that nearest prior work does not already cover.

Only include gaps with plausible breakthrough potential >= {request.ambition_floor}/10. If
you cannot find enough, return fewer gaps and explain the shortage in the risk/evidence fields.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""


def idea_generation_prompt(
    request: ResearchRequest, literature: str, gaps: list[NoveltyGap]
) -> str:
    gap_text = "\n".join(f"- {gap.gap} | why now: {gap.why_now}" for gap in gaps)
    schema = {
        "ideas": [
            {
                "title": "Concrete paper title.",
                "thesis": "One-sentence central claim.",
                "core_mechanism": "The new method, benchmark, theory, or evaluation mechanism.",
                "novelty_claim": "Why this is not just prior work plus engineering.",
                "why_reviewers_care": "Significance argument.",
                "nearest_prior_work": ["Closest paper/system this must beat or differ from."],
                "decisive_differentiator": "The one technical move that makes this a new paper, not a variant.",
                "innovation_score": 8,
                "paper_worth_score": 8,
                "venue_upside_score": 8,
                "fixed_pool_only_score": 5,
                "main_conference_checklist": [
                    {
                        "name": "Originality against nearest prior work",
                        "score": 8,
                        "threshold": 8,
                        "evidence_needed": "Concrete evidence required.",
                        "failure_mode": "What would fail review.",
                    }
                ],
                "expected_evidence": ["Main result reviewers would need."],
                "first_experiments": ["Specific first experiment or ablation."],
                "key_risks": ["Potential rejection reason."],
                "citations": [{"title": "Source title", "url": "https://...", "note": "Why it matters"}],
            }
        ]
    }
    return f"""You are the Idea Generator.

{request_context(request)}

Literature map:
{literature}

Novelty gaps:
{gap_text}

{main_conference_checklist_instruction(request.ambition_floor)}

Generate exactly {request.idea_count} original paper candidates, but do not pad the list with
incremental ideas. If necessary, create fewer but stronger candidates.

Hard bar:
- innovation_score >= {request.ambition_floor}
- paper_worth_score >= {request.ambition_floor}
- venue_upside_score >= {request.ambition_floor}
- fixed_pool_only_score >= {request.ambition_floor}; the first isolating MVP should be strong
  enough to matter on its own, not merely serve as a weak prelude to closed-loop scale.

Retry-feedback handling:
- If the constraints include failed-batch or reviewer feedback, treat it as binding.
- Do not merely avoid previous titles. Avoid the underlying failure modes: thin novelty,
  baseline vulnerability, weak fixed-pool MVP, unclear mechanism, or a story that would top out
  at 7/10.
- Generate candidates that directly repair the listed selector risks, novelty blockers, and
  reviewer fatal flaws with a different technical mechanism. If a candidate cannot explain why
  it clears the previous feedback, discard it internally and make a stronger one.

Each candidate must contain a technical mechanism that changes the research object, not only
the evaluation schedule. It must identify nearest prior work and name the decisive differentiator.
Each candidate must include the full main_conference_checklist with all checklist dimensions.
Prefer high-upside ideas that could plausibly become a new subproblem, new optimization paradigm,
new benchmark with a strong method attached, or new theory-backed algorithm.

Do not propose:
- minor prompt search variants,
- generic wrappers around GEPA/MIPROv2/CAPO/Hyperband,
- ideas whose main win is more compute, more calls, ensembling, hidden supervision, or better prompts,
- ideas whose novelty depends on a literature oversight likely to disappear under reviewer search.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""


def review_prompt(
    request: ResearchRequest,
    ideas: list[IdeaCandidate],
    novelty_audits: list[IdeaNoveltyAudit],
    literature: str,
) -> str:
    ideas_json = json.dumps([idea.model_dump(mode="json") for idea in ideas], indent=2)
    novelty_audits_json = json.dumps(
        [audit.model_dump(mode="json") for audit in novelty_audits], indent=2
    )
    schema = {
        "reviews": [
            {
                "idea_title": "Must match a candidate title.",
                "reviewer": "ICLR Reviewer 2|NeurIPS Area Chair|Methods Reviewer|Empiricism Reviewer",
                "novelty": 1,
                "significance": 1,
                "correctness": 1,
                "feasibility": 1,
                "clarity": 1,
                "research_worth_score": 1,
                "paper_worth_score": 1,
                "venue_upside_score": 1,
                "fixed_pool_only_score": 1,
                "main_conference_checklist": [
                    {
                        "name": "Originality against nearest prior work",
                        "score": 8,
                        "threshold": 8,
                        "evidence_needed": "Concrete evidence required.",
                        "failure_mode": "What would fail review.",
                    }
                ],
                "recommendation": "strong_reject|reject|borderline|accept|strong_accept",
                "fatal_flaws": ["Most dangerous objection."],
                "required_experiments": ["Experiment needed before submission."],
                "rescue_moves": ["Concrete fix that could change the review."],
            }
        ]
    }
    return f"""You are the Skeptical Review Board. Act like demanding ICLR and NeurIPS reviewers.

{request_context(request)}

Literature map:
{literature}

Candidate ideas:
{ideas_json}

Novelty collision audit:
{novelty_audits_json}

{main_conference_checklist_instruction(request.ambition_floor)}

Score each idea using novelty, significance, correctness, feasibility, and clarity from 1 to 10.
Also score:
- research_worth_score: should a serious researcher spend weeks on this?
- paper_worth_score: would a positive MVP justify drafting a paper?
- venue_upside_score: could the full result plausibly clear ICLR/NeurIPS?
- fixed_pool_only_score: venue upside if the only positive evidence is fixed-pool reranking.

Be harsh. A score of 8 means the idea has a real path to a strong submission, not just a
reasonable workshop paper. Penalize incremental scheduler/wrapper ideas unless the mechanism is
clearly new and the comparison story is decisive.
Use the novelty collision audit as hard evidence. If the audit marks an idea borderline or fail,
do not give it accept-level scores unless you can explain the reframing or decisive experiment
that genuinely resolves the collision.
For every review, include a full main_conference_checklist. Any checklist dimension below
{request.ambition_floor}/10 should appear in fatal_flaws or rescue_moves.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""


def novelty_collision_prompt(
    request: ResearchRequest, literature: str, ideas: list[IdeaCandidate]
) -> str:
    ideas_json = json.dumps([idea.model_dump(mode="json") for idea in ideas], indent=2)
    schema = {
        "audits": [
            {
                "idea_title": "Must exactly match a candidate title.",
                "closest_prior_work": [
                    {
                        "prior_work": "Exact paper/system name.",
                        "url": "https://...",
                        "overlap": "What part of the candidate this prior work already covers.",
                        "severity": "low|medium|high|fatal",
                        "what_to_cede": "Claim the new paper should not make anymore.",
                        "required_differentiator": "What must be shown to stay novel.",
                    }
                ],
                "novelty_score": 1,
                "paper_worth_score": 1,
                "venue_upside_score": 1,
                "main_track_verdict": "pass|borderline|fail",
                "workshop_risk": True,
                "main_track_blockers": ["Why reviewers would reject this as not main-track."],
                "required_reframing": ["How to frame the idea if it remains viable."],
                "direct_comparisons_required": ["Specific method/paper that must be compared."],
            }
        ],
        "batch_verdict": "pass|borderline|fail",
        "batch_rationale": "Whether any idea truly clears the main-track novelty bar.",
    }
    return f"""You are the Novelty Collision Auditor.

Your task is to protect the user from spending weeks on an idea that is good but not
main-conference-worthy. Use web search aggressively. Search for each candidate's core mechanism,
not only its title. Look for very recent arXiv, OpenReview, conference, benchmark, and GitHub work
that a reviewer would cite as "already done", "incremental", or "workshop-level".

{request_context(request)}

Literature map:
{literature}

Candidate ideas:
{ideas_json}

Audit standard:
- Treat the ambition floor as {request.ambition_floor}/10.
- A "pass" means the idea has a defensible main-track novelty claim even after closest-prior-work search.
- A "borderline" means the idea is interesting but likely workshop/borderline unless the framing or mechanism changes.
- A "fail" means nearest prior work already owns the headline, or the contribution is only a stronger engineering version.
- Penalize vague differentiators such as "more rigorous", "better prompt", "better benchmark", or "uses an LLM".
- Reward only crisp differentiators such as a new formal object, new optimization target, new identification strategy,
  new benchmark plus method, new theory, or a decisive empirical phenomenon prior work did not test.

For every candidate:
1. Identify the closest novelty collisions, including 2025-2026 papers if relevant.
2. State what claims must be ceded.
3. State the exact reframing required to stay publishable.
4. List mandatory direct comparisons.
5. Give a main-track verdict and provisional scores.

Important: the final novelty_score, paper_worth_score, and venue_upside_score will be
assigned by three independent score-auditor subagents. Your score fields are fallback values
only. Focus most of your effort on collision discovery, blockers, required reframing, and
mandatory comparisons.

If no candidate passes, say so. Do not rescue the batch by optimism.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""


def novelty_dimension_score_prompt(
    request: ResearchRequest,
    literature: str,
    ideas: list[IdeaCandidate],
    novelty_audits: list[IdeaNoveltyAudit],
    *,
    score_field: str,
) -> str:
    ideas_json = json.dumps([idea.model_dump(mode="json") for idea in ideas], indent=2)
    novelty_audits_json = json.dumps(
        [audit.model_dump(mode="json") for audit in novelty_audits], indent=2
    )
    dimension_guidance = {
        "novelty_score": {
            "agent": "Novelty Score Auditor",
            "definition": (
                "Score only the originality and defensibility of the novelty claim after "
                "accounting for the listed closest prior-work collisions."
            ),
            "ignore": (
                "Do not reward feasibility, paper polish, implementation convenience, or broad "
                "community usefulness unless they change the novelty claim."
            ),
        },
        "paper_worth_score": {
            "agent": "Paper Worth Score Auditor",
            "definition": (
                "Score whether a positive MVP would justify writing a serious paper, given "
                "the method, baselines, ablations, risks, and collision audit."
            ),
            "ignore": (
                "Do not score raw novelty alone. A novel idea can still be a weak paper if "
                "the MVP is thin, the baselines are impossible to beat, or the result would "
                "not teach reviewers much."
            ),
        },
        "venue_upside_score": {
            "agent": "Venue Upside Score Auditor",
            "definition": (
                "Score the ceiling for ICLR/NeurIPS main-conference upside if the project is "
                "executed rigorously and the key experiment is positive."
            ),
            "ignore": (
                "Do not score short-term feasibility alone. Reward breadth, significance, "
                "mechanistic insight, reviewer-survivability, and potential to define a "
                "recognizable research contribution."
            ),
        },
    }
    if score_field not in dimension_guidance:
        raise ValueError(f"Unknown novelty dimension score field: {score_field}")
    guidance = dimension_guidance[score_field]
    schema = {
        "scores": [
            {
                "idea_title": "Must exactly match a candidate title.",
                "score": 1,
                "rationale": "Short reason for this dimension score.",
                "risks": ["Dimension-specific reason the score is not higher."],
            }
        ]
    }
    return f"""You are the {guidance["agent"]}.

You are one of three independent score auditors. Your only job is to assign
`{score_field}`. Do not assign or discuss the other two novelty-audit scores.

{request_context(request)}

Literature map:
{literature}

Candidate ideas:
{ideas_json}

Collision audit from the separate Novelty Collision Auditor:
{novelty_audits_json}

Scoring mandate:
{guidance["definition"]}

What to ignore:
{guidance["ignore"]}

Scoring standard:
- Use the ambition floor {request.ambition_floor}/10 as a real main-conference bar.
- A 7 means promising but still borderline.
- An 8 means a defensible main-track path if the decisive experiment succeeds.
- A 9 means unusually strong and hard for reviewers to dismiss if executed well.
- A 10 should be reserved for rare, field-shaping ideas.
- Be willing to score below 7 when the collision audit or MVP story is weak.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""


def experiment_design_prompt(
    request: ResearchRequest,
    selected_idea: IdeaCandidate,
    reviews_json: str,
    literature: str,
    selection: SelectionDecision,
) -> str:
    idea_json = json.dumps(selected_idea.model_dump(mode="json"), indent=2)
    selection_json = json.dumps(selection.model_dump(mode="json"), indent=2)
    return f"""You are the Experiment Designer.

{request_context(request)}

Literature map:
{literature}

Selected idea:
{idea_json}

Selection decision:
{selection_json}

Skeptical reviews:
{reviews_json}

Design the strongest executable validation plan for the selected idea.

Return Markdown with:
1. A baseline matrix: method, dataset/task, metric, expected reviewer objection answered.
2. Critical ablations for the selected idea.
3. Minimum viable experiment that can be run first.
4. Compute/data assumptions.
5. Reproducibility checklist.
6. Reviewer-risk mitigation plan.

Be specific enough that a researcher can start implementation immediately.
"""


def selection_prompt(
    request: ResearchRequest,
    literature: str,
    ideas: list[IdeaCandidate],
    novelty_audits: list[IdeaNoveltyAudit],
    reviews_json: str,
) -> str:
    ideas_json = json.dumps([idea.model_dump(mode="json") for idea in ideas], indent=2)
    novelty_audits_json = json.dumps(
        [audit.model_dump(mode="json") for audit in novelty_audits], indent=2
    )
    schema = {
        "selected_title": "Must exactly match one candidate title.",
        "rationale": "Why this is the best research bet.",
        "iclr_neurips_case": "Why the idea could be worth ICLR/NeurIPS if executed well.",
        "research_worth_score": 8,
        "paper_worth_score": 8,
        "venue_upside_score": 8,
        "fixed_pool_only_score": 5,
        "main_conference_checklist": [
            {
                "name": "Originality against nearest prior work",
                "score": 8,
                "threshold": 8,
                "evidence_needed": "Concrete evidence required.",
                "failure_mode": "What would fail review.",
            }
        ],
        "breakthrough_condition": "The minimum result that would make this a real ICLR/NeurIPS paper.",
        "decisive_strengths": ["Specific strengths."],
        "decisive_risks": ["Specific risks."],
        "required_next_steps": ["What must be done before writing the paper."],
        "score": 1,
    }
    return f"""You are the Best Idea Selector.

Your job is to choose exactly one idea that is most worth researching and most likely to become a serious ICLR/NeurIPS submission if executed rigorously.

{request_context(request)}

Literature map:
{literature}

Candidate ideas:
{ideas_json}

Review board output:
{reviews_json}

Novelty collision audit:
{novelty_audits_json}

{main_conference_checklist_instruction(request.ambition_floor)}

Selection criteria:
- strongest novelty relative to nearest prior work,
- clearest path to SOTA or meaningful benchmark contribution,
- feasible implementation and evaluation,
- fair head-to-head comparison potential,
- clear story reviewers will understand,
- rejection risks that can be mitigated with experiments.

Do not choose the flashiest idea. Choose the best paper bet above the ambition floor.

Selection gate:
- research_worth_score must be >= {request.ambition_floor}
- paper_worth_score must be >= {request.ambition_floor}
- venue_upside_score must be >= {request.ambition_floor}
- fixed_pool_only_score must be >= {request.ambition_floor}
- every main_conference_checklist item should score >= {request.ambition_floor}; if any item
  is below the floor, the selected idea has not cleared the main-conference bar.
- novelty-audit N/P/V scores are hard evidence, but they do not by themselves clear the
  selector gate; the selector dimensions and checklist must also clear the floor.
- if no idea clears the full gate, still select the highest-upside idea and clearly state
  that the batch failed the ambition floor.
- Do not collapse every failed-gate selection to 6/10. Assign a score that reflects the
  selected idea's defensible overall paper promise: use 7-8 for strong near-misses with
  high novelty/upside but repairable gate gaps, and reserve <=6 for salvage-only, weak,
  or brittle directions.
- if the selected idea fails any required selector dimension or checklist row, score must
  stay below the ambition floor until those gaps are fixed.
- never select an idea with novelty audit verdict "fail" unless all ideas fail; in that case set score <= 5.
- selecting a "borderline" idea requires an explicit reframing plan and score <= 7 until the reframing is validated.
- if the novelty audit found high or fatal collisions, the rationale must explain exactly what claims are ceded
  and what differentiator remains.

Prefer super-innovative directions over merely feasible ones, but do not fake certainty. The
selected idea must have a named breakthrough condition: the concrete empirical or theoretical
result that would move it from promising to submission-grade.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""


def implementation_doc_prompt(
    request: ResearchRequest,
    selected_idea: IdeaCandidate,
    selection: SelectionDecision,
    novelty_audits: list[IdeaNoveltyAudit],
    literature: str,
    reviews_json: str,
    experiment_plan: str,
) -> str:
    idea_json = json.dumps(selected_idea.model_dump(mode="json"), indent=2)
    selection_json = json.dumps(selection.model_dump(mode="json"), indent=2)
    novelty_audits_json = json.dumps(
        [audit.model_dump(mode="json") for audit in novelty_audits], indent=2
    )
    return f"""You are the Implementation Architect for a research paper project.

Create a detailed, practical implementation document for the selected idea. This document will be exported to DOCX and used by a researcher to build the method and write the paper.

{request_context(request)}

Selected idea:
{idea_json}

Selection decision:
{selection_json}

Novelty collision audit:
{novelty_audits_json}

Relevant literature map:
{literature}

Reviewer objections:
{reviews_json}

Experiment plan:
{experiment_plan}

Write a detailed Markdown document with these sections:
0. One-page execution brief: the MVP build path, the breakthrough condition, and the kill criteria.
1. Paper thesis and contribution claim.
1a. Novelty collision diff: closest prior work, what to cede, what to claim, and mandatory direct comparisons.
2. Exact method design, including algorithm steps and pseudocode.
3. How to implement the method in code: modules, classes, data structures, APIs, and logging.
4. Fair comparison plan against the SOTA baselines named in the request.
5. Benchmarks, datasets, metrics, cost accounting, and statistical tests.
6. Ablations and negative controls.
7. Failure modes and mitigation.
8. Week-by-week execution plan for 6 weeks.
9. Paper outline with section-by-section writing guidance.
10. Reproducibility checklist and artifact-release plan.
11. Acceptance-risk checklist for ICLR/NeurIPS.

Make it concrete. Prefer implementation details over generic advice. If the request names specific baselines, include one-on-one reproduction and comparison details for them.
Do not bury the first implementation step. The first two pages must make it obvious what to build
first, what result is needed for a paper, and what result should kill or pivot the idea.
If the novelty audit is borderline or fail, do not write as if acceptance is likely; write the required
reframing and the proof needed before this should be treated as a main-paper project.
"""


def final_synthesis_prompt(
    request: ResearchRequest,
    literature: str,
    gaps: list[NoveltyGap],
    ideas: list[IdeaCandidate],
    novelty_audits: list[IdeaNoveltyAudit],
    reviews_json: str,
    selection: SelectionDecision,
    experiment_plan: str,
    implementation_plan: str,
) -> str:
    ideas_json = json.dumps([idea.model_dump(mode="json") for idea in ideas], indent=2)
    gaps_json = json.dumps([gap.model_dump(mode="json") for gap in gaps], indent=2)
    novelty_audits_json = json.dumps(
        [audit.model_dump(mode="json") for audit in novelty_audits], indent=2
    )
    return f"""You are the Chief Scientist.

{request_context(request)}

Literature map:
{literature}

Novelty gaps:
{gaps_json}

Ideas:
{ideas_json}

Novelty collision audit:
{novelty_audits_json}

Review board output:
{reviews_json}

Best idea selection:
{json.dumps(selection.model_dump(mode="json"), indent=2)}

Experiment design:
{experiment_plan}

Detailed implementation plan:
{implementation_plan}

Write the final research strategy in Markdown:
1. Rank the ideas.
2. Name the selected best idea and explain why it survives review pressure.
3. State the exact paper contribution.
4. List the must-cite prior work categories.
5. Give the first 14-day execution plan.
6. Give the kill criteria: what result would tell us to abandon or pivot.
7. Include an LLM-usage disclosure note suitable for a paper draft.

Be direct and realistic. Do not promise acceptance. Optimize for ideas that can become rigorous submissions.
"""
