from research_foundry.utils import parse_jsonish


def test_parse_jsonish_handles_fenced_json():
    parsed = parse_jsonish(
        """```json
{"ideas": [{"title": "A"}]}
```"""
    )

    assert parsed["ideas"][0]["title"] == "A"


def test_parse_jsonish_handles_prose_wrapped_json():
    parsed = parse_jsonish('Here is the JSON: {"gaps": [{"gap": "x"}]} done.')

    assert parsed["gaps"][0]["gap"] == "x"

