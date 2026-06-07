import pytest

from research_foundry.config import Settings
from research_foundry.gateway import (
    OpenAIResponsesGateway,
    extract_response_text,
    reasoning_effort_for_model,
    web_search_tool,
)


def test_extract_response_text_from_response_shape():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello"},
                    {"type": "output_text", "text": "world"},
                ],
            }
        ]
    }

    assert extract_response_text(response) == "hello\nworld"


def test_deep_web_search_tool_uses_medium_context():
    tool = web_search_tool(deep=True)

    assert tool["type"] == "web_search"
    assert tool["search_context_size"] == "medium"
    assert "return_token_budget" not in tool


def test_gpt_55_models_force_high_reasoning():
    assert reasoning_effort_for_model("gpt-5.5") == "high"
    assert reasoning_effort_for_model("gpt-5.5-pro") == "high"
    assert reasoning_effort_for_model("gpt-5.5-pro", "low") == "high"
    assert reasoning_effort_for_model("gpt-5", "medium") == "medium"
    assert reasoning_effort_for_model("o3-deep-research") is None


class _FakeResponses:
    def __init__(self):
        self.kwargs = None
        self.create_count = 0

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.create_count += 1
        return {
            "id": "resp_fake",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                }
            ],
        }


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


@pytest.mark.asyncio
async def test_gateway_builds_response_request_with_metadata_and_tools():
    client = _FakeClient()
    gateway = OpenAIResponsesGateway(Settings(openai_api_key="sk-test"), client=client)
    tool = web_search_tool(deep=True)

    artifact = await gateway.run_text(
        agent_name="Literature Cartographer",
        prompt="scan",
        model="o3-deep-research",
        tools=[tool],
        background=True,
        output_kind="literature",
    )

    assert artifact.content == "done"
    assert client.responses.kwargs["model"] == "o3-deep-research"
    assert client.responses.kwargs["tools"] == [tool]
    assert client.responses.kwargs["background"] is True
    assert "reasoning" not in client.responses.kwargs
    assert client.responses.kwargs["metadata"]["agent_name"] == "Literature Cartographer"


@pytest.mark.asyncio
async def test_gateway_forces_gpt_55_reasoning_high_without_explicit_effort():
    client = _FakeClient()
    gateway = OpenAIResponsesGateway(Settings(openai_api_key="sk-test"), client=client)

    await gateway.run_text(
        agent_name="Novelty Gap Miner",
        prompt="mine gaps",
        model="gpt-5.5",
        output_kind="gaps",
    )

    assert client.responses.kwargs["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_gateway_forces_gpt_55_pro_reasoning_high_over_lower_effort():
    client = _FakeClient()
    gateway = OpenAIResponsesGateway(Settings(openai_api_key="sk-test"), client=client)

    await gateway.run_text(
        agent_name="Skeptical Review Board",
        prompt="review",
        model="gpt-5.5-pro",
        reasoning_effort="low",
        output_kind="reviews",
    )

    assert client.responses.kwargs["reasoning"] == {"effort": "high"}


class _RetryThenSuccessResponses:
    def __init__(self):
        self.create_count = 0

    async def create(self, **kwargs):
        self.create_count += 1
        if self.create_count == 1:
            return {
                "id": "resp_failed",
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "An error occurred while processing your request.",
                },
            }
        return {
            "id": "resp_ok",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "recovered"}],
                }
            ],
        }


class _RetryThenSuccessClient:
    def __init__(self):
        self.responses = _RetryThenSuccessResponses()


@pytest.mark.asyncio
async def test_gateway_retries_terminal_server_error_response():
    client = _RetryThenSuccessClient()
    settings = Settings(
        openai_api_key="sk-test",
        response_max_attempts=2,
        response_retry_base_seconds=0,
    )
    gateway = OpenAIResponsesGateway(settings, client=client)

    artifact = await gateway.run_text(
        agent_name="Literature Cartographer",
        prompt="scan",
        model="o3-deep-research",
        output_kind="literature",
    )

    assert artifact.content == "recovered"
    assert artifact.response_id == "resp_ok"
    assert artifact.metadata["attempts"] == 2
    assert client.responses.create_count == 2


class _NonRetryableFailedResponses:
    def __init__(self):
        self.create_count = 0

    async def create(self, **kwargs):
        self.create_count += 1
        return {
            "id": "resp_bad_request",
            "status": "failed",
            "error": {
                "code": "invalid_request_error",
                "message": "The request is invalid.",
            },
        }


class _NonRetryableFailedClient:
    def __init__(self):
        self.responses = _NonRetryableFailedResponses()


@pytest.mark.asyncio
async def test_gateway_does_not_retry_non_transient_terminal_failure():
    client = _NonRetryableFailedClient()
    settings = Settings(
        openai_api_key="sk-test",
        response_max_attempts=3,
        response_retry_base_seconds=0,
    )
    gateway = OpenAIResponsesGateway(settings, client=client)

    with pytest.raises(RuntimeError, match="invalid_request_error"):
        await gateway.run_text(
            agent_name="Literature Cartographer",
            prompt="scan",
            model="o3-deep-research",
            output_kind="literature",
        )

    assert client.responses.create_count == 1


class _RetryablePollingError(Exception):
    status_code = 500


class _PollingRetryResponses:
    def __init__(self):
        self.create_count = 0
        self.retrieve_count = 0

    async def create(self, **kwargs):
        self.create_count += 1
        return {"id": "resp_polling", "status": "queued"}

    async def retrieve(self, response_id):
        assert response_id == "resp_polling"
        self.retrieve_count += 1
        if self.retrieve_count == 1:
            raise _RetryablePollingError("temporary retrieve failure")
        return {
            "id": "resp_polling",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "polled"}],
                }
            ],
        }


class _PollingRetryClient:
    def __init__(self):
        self.responses = _PollingRetryResponses()


@pytest.mark.asyncio
async def test_gateway_retries_polling_without_starting_new_response():
    client = _PollingRetryClient()
    settings = Settings(
        openai_api_key="sk-test",
        max_wait_seconds=1,
        poll_seconds=0,
        response_retry_base_seconds=0,
    )
    gateway = OpenAIResponsesGateway(settings, client=client)

    artifact = await gateway.run_text(
        agent_name="Literature Cartographer",
        prompt="scan",
        model="o3-deep-research",
        output_kind="literature",
    )

    assert artifact.content == "polled"
    assert client.responses.create_count == 1
    assert client.responses.retrieve_count == 2
