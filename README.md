# Research Foundry

Research Foundry is a multi-agent system for generating, stress-testing, selecting, and operationalizing ML research ideas. It uses OpenAI deep research models for literature review, then routes the result through novelty mining, idea generation, reviewer simulation, best-idea selection, experiment design, and final paper-strategy synthesis.

The first implementation is intentionally pipeline-first: the orchestrator owns state, provenance, saved artifacts, and evaluation contracts. An optional OpenAI Agents SDK team builder is included for interactive agent workflows.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For live runs:

```bash
export OPENAI_API_KEY="sk-..."
```

## Dry Run

Dry run exercises the full system without making API calls:

```bash
research-foundry run \
  --field "multi-agent systems for scientific discovery" \
  --objective "find original ideas likely to survive ICLR or NeurIPS review" \
  --constraint "must include executable evaluation plans" \
  --ideas 3 \
  --dry-run
```

## Live Deep Research Run

```bash
research-foundry run \
  --field "LLM agents for automated ML research" \
  --objective "generate novel, feasible ICLR/NeurIPS paper ideas" \
  --constraint "must identify must-cite prior work and strong baselines" \
  --constraint "prefer ideas that can be validated in 4-8 weeks" \
  --ideas 3
```

Use `--fast` to switch literature research from `o3-deep-research` to `o4-mini-deep-research`.

Background deep-research responses can run longer than 15 minutes. The CLI now
waits up to 3600 seconds by default before timing out a background Responses API
stage. Use `--max-wait-seconds 7200` for especially broad literature scans, or
set `RESEARCH_FOUNDRY_MAX_WAIT_SECONDS`.

Live Responses API calls retry transient failures such as terminal
`status=failed` responses with `server_error` before the pipeline gives up. By
default the gateway tries 3 attempts with exponential backoff starting at 2
seconds. Tune this with `RESEARCH_FOUNDRY_RESPONSE_MAX_ATTEMPTS` and
`RESEARCH_FOUNDRY_RESPONSE_RETRY_BASE_SECONDS`.

By default, the CLI keeps generating fresh idea batches until the Best Idea
Selector clears the configured `--ambition-floor`, or until 3 batches are
exhausted. This selector gate requires the headline selector score, the selector
dimension scores, the fixed-pool-only score, and every main-conference checklist
row to clear the floor. A rejected or near-miss selector decision is saved as
feedback for the next batch, but downstream experiment design, implementation
planning, and final synthesis are skipped for that failed batch.

Use `--until-novelty-pass` when you also do not want the run to stop on a batch
where no idea clears the novelty-collision audit. With no value it tries up to 3
fresh batches; with a value it tries up to at least that many batches, subject to
the default selector retry budget:

```bash
research-foundry run \
  --field "API-efficient prompt optimization" \
  --objective "find a main-track-worthy method" \
  --ideas 5 \
  --ambition-floor 8 \
  --until-novelty-pass 5
```

Use `--until-selector-score` when you want the selector gate to require a
headline selector score above the ambition floor. The full selector gate still
has to clear all dimension and checklist floors. The effective threshold is
never below `--ambition-floor`. With no value it uses
`max(8/10, --ambition-floor)` and tries up to 3 batches:

```bash
research-foundry run \
  --field "API-efficient prompt optimization" \
  --objective "find a main-track-worthy method" \
  --ideas 5 \
  --ambition-floor 8 \
  --until-selector-score
```

The two retry gates can be combined. For example, `--until-novelty-pass 5
--until-selector-score` stops only when a batch has at least one novelty-audit
pass and the selected idea scores at least `8/10`, or when 5 batches are exhausted.

Each failed batch is saved, then the next batch receives an explicit constraint
to avoid repeating, renaming, merging, or lightly modifying the failed/borderline
directions and low-scoring selected ideas. The retry constraint also includes a
compact reviewer-feedback memo: selector score gaps, decisive risks, required
repair moves, novelty blockers, closest collisions, demanded differentiators,
review fatal flaws, and rescue moves. The next Idea Generator pass is instructed
to repair those failure modes directly, not merely sample a different title that
still tops out at 7/10. If no batch clears the requested retry gates, the latest
output should be treated as a pivot document rather than a cleared main-paper
plan.

## Terminal Display

The CLI uses Rich and tqdm by default:

- Rich prints colored run headers, stage logs, artifact paths, and final selection panels.
- tqdm shows the live 9-stage pipeline bar.
- The full generated idea list is printed as soon as the Idea Generator stage finishes.
- The novelty-collision audit is printed before reviews and selection, so borderline ideas are visible early.
- The selected idea is printed before any downstream planning starts. If it does
  not clear the selector gate, downstream planning is skipped and the next batch
  is generated.
- Long stages refresh every second with elapsed time, so deep research does not look frozen.

Use `--no-progress` if you need quieter output for CI logs.

## Output

Each accepted run writes:

- `runs/<run_id>/report.md`
- `runs/<run_id>/report.json`
- `runs/<run_id>/selected_idea_implementation_plan.docx`
- `runs/<run_id>/artifacts/*.md`

Rejected retry batches write `report.md`, `report.json`, and raw artifacts, but
skip `selected_idea_implementation_plan.docx` because no implementation plan has
been approved.

## Architecture

Agents:

- `Literature Cartographer`: deep research over current literature.
- `Novelty Gap Miner`: finds underexplored assumptions and missing evidence.
- `Idea Generator`: turns gaps into concrete paper candidates.
- `Novelty Collision Auditor`: searches for closest prior-work collisions and marks each idea pass, borderline, or fail for main-track novelty.
- `Novelty Score Auditor`: independently assigns the novelty score shown as `N`.
- `Paper Worth Score Auditor`: independently assigns the paper-worth score shown as `P`.
- `Venue Upside Score Auditor`: independently assigns the venue-upside score shown as `V`.
- `Skeptical Review Board`: simulates ICLR/NeurIPS reviewer objections.
- `Best Idea Selector`: chooses the single most research-worthy and ICLR/NeurIPS-plausible idea.
- `Experiment Designer`: strengthens the selected idea with baselines, ablations, metrics, and reproducibility checks.
- `Implementation Architect`: creates a detailed implementation plan exported to DOCX.
- `Chief Scientist`: synthesizes the final recommendation.

Every agent stage receives a Responses API web-search tool. The literature stage uses deep research web search; downstream stages use standard web search so they can verify current baselines, benchmarks, and reviewer-risk details while working.

Default model routing:

- Deep research: `o3-deep-research`
- Fast deep research: `o4-mini-deep-research`
- Synthesis: `gpt-5.5` with `reasoning.effort = high`
- Review: `gpt-5.5-pro` with `reasoning.effort = high`

The gateway also enforces `reasoning.effort = high` for any direct `gpt-5.5` or `gpt-5.5-pro` call, even if a future caller forgets to set it.

## Notes

This system can help generate and evaluate research directions, but it cannot guarantee conference acceptance. Treat it as a ruthless thought partner: it should surface stronger ideas, sharper baselines, and earlier failure modes.
