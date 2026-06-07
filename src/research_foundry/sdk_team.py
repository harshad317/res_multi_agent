from __future__ import annotations

from typing import Any, Awaitable, Callable

from research_foundry.config import Settings
from research_foundry.gateway import requires_high_reasoning


DeepResearchCallable = Callable[[str], Awaitable[str]]


def _model_settings_for(model: str) -> Any:
    try:
        from agents import ModelSettings
    except ImportError:
        return None

    if requires_high_reasoning(model):
        return ModelSettings(reasoning={"effort": "high"})
    return ModelSettings()


def build_openai_agents_team(
    settings: Settings,
    *,
    deep_research_scan: DeepResearchCallable,
) -> Any:
    """Build an optional OpenAI Agents SDK team.

    The pipeline in `ResearchFoundry` is the production path because it keeps
    artifacts structured and easy to evaluate. This builder exposes the same
    roles through the Agents SDK for interactive workflows.
    """

    try:
        from agents import Agent, function_tool
    except ImportError as exc:
        raise RuntimeError(
            "openai-agents is not installed. Run `pip install openai-agents` "
            "or use the pipeline/CLI without the optional SDK team."
        ) from exc

    @function_tool
    async def run_deep_literature_scan(query: str) -> str:
        """Run an OpenAI deep research literature scan for the supplied query."""
        return await deep_research_scan(query)

    literature = Agent(
        name="Literature Cartographer",
        handoff_description="Deep literature research and must-cite prior-work mapping.",
        instructions=(
            "Map the current literature, identify must-cite baselines, and flag "
            "incremental danger zones. Never fabricate citations."
        ),
        model=settings.frontier_model,
        model_settings=_model_settings_for(settings.frontier_model),
        tools=[run_deep_literature_scan],
    )

    gap_miner = Agent(
        name="Novelty Gap Miner",
        handoff_description="Finds non-obvious, review-survivable research gaps.",
        instructions=(
            "Find technically crisp gaps that differ from prior work and can be "
            "validated with convincing experiments."
        ),
        model=settings.frontier_model,
        model_settings=_model_settings_for(settings.frontier_model),
    )

    reviewer = Agent(
        name="Skeptical Review Board",
        handoff_description="Simulates harsh ICLR and NeurIPS review pressure.",
        instructions=(
            "Review ideas harshly for novelty, significance, correctness, "
            "feasibility, clarity, missing baselines, and fatal flaws."
        ),
        model=settings.reviewer_model,
        model_settings=_model_settings_for(settings.reviewer_model),
    )

    experiment_designer = Agent(
        name="Experiment Designer",
        handoff_description="Turns ideas and reviewer objections into executable validation plans.",
        instructions=(
            "Design baseline matrices, ablations, minimum viable experiments, "
            "compute assumptions, and reproducibility plans."
        ),
        model=settings.frontier_model,
        model_settings=_model_settings_for(settings.frontier_model),
    )

    chief = Agent(
        name="Chief Scientist",
        instructions=(
            "Coordinate the research team. Use specialists as bounded tools, "
            "then synthesize a rigorous final research strategy."
        ),
        model=settings.frontier_model,
        model_settings=_model_settings_for(settings.frontier_model),
        tools=[
            literature.as_tool(
                tool_name="map_literature",
                tool_description="Run deep literature mapping for a research direction.",
            ),
            gap_miner.as_tool(
                tool_name="mine_novelty_gaps",
                tool_description="Find review-survivable novelty gaps from literature.",
            ),
            reviewer.as_tool(
                tool_name="review_research_ideas",
                tool_description="Stress-test candidate ideas like ICLR/NeurIPS reviewers.",
            ),
            experiment_designer.as_tool(
                tool_name="design_experiments",
                tool_description="Create executable experiments and ablations for candidate ideas.",
            ),
        ],
    )
    return chief
