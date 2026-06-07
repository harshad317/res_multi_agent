from research_foundry.config import DEFAULT_MAX_WAIT_SECONDS, Settings


def test_default_background_response_wait_is_long_enough_for_deep_research(monkeypatch):
    monkeypatch.delenv("RESEARCH_FOUNDRY_MAX_WAIT_SECONDS", raising=False)

    settings = Settings(openai_api_key="sk-test")

    assert DEFAULT_MAX_WAIT_SECONDS == 3600
    assert settings.max_wait_seconds == 3600


def test_background_response_wait_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_FOUNDRY_MAX_WAIT_SECONDS", "7200")

    settings = Settings(openai_api_key="sk-test")

    assert settings.max_wait_seconds == 7200
