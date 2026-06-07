from research_foundry.sdk_team import _model_settings_for


def test_agents_sdk_model_settings_force_high_reasoning_for_gpt_55_models():
    frontier_settings = _model_settings_for("gpt-5.5")
    reviewer_settings = _model_settings_for("gpt-5.5-pro")

    assert frontier_settings.reasoning.effort == "high"
    assert reviewer_settings.reasoning.effort == "high"


def test_agents_sdk_model_settings_do_not_force_other_models():
    settings = _model_settings_for("o3-deep-research")

    assert settings.reasoning is None
