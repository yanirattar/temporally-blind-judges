import json

import pytest

from tbj.judge import (SCHEMAS, Job, ParseError, build_request, cost_usd, job_key, load_prompt, pair_order,
                       parse_answer, usage_from)

SETTINGS = {"model": "gemini-3.8-flash", "media_resolution": "MEDIA_RESOLUTION_LOW",
            "media_processing": "STATIC", "thinking_level": "LOW", "temperature": None}
COND = {"sha256": "a" * 64, "rev_sha256": "b" * 64}
SINGLE = {"kind": "single", "prompt": "default_v1", "fps": 1}
DENSE = {"kind": "single", "prompt": "default_v1", "fps": 4}
PAIRED = {"kind": "paired", "prompt": "paired_v1", "fps": 1}
CHECK = {"kind": "checklist", "prompt": "checklist_v1", "fps": 1}


def test_prompts_load_and_schemas_list_reason_first():
    for name in ("default_v1", "checklist_v1", "paired_v1"):
        assert len(load_prompt(name).text) > 200
    for schema in SCHEMAS.values():
        assert list(schema["properties"])[0] == "reason"
        assert set(schema["required"]) == set(schema["properties"])


def test_single_request_puts_video_before_text():
    req, meta = build_request(Job("c", "default"), SINGLE, load_prompt("default_v1"), SETTINGS, "uri-x")
    parts = req["contents"][0]["parts"]
    assert parts[0]["file_data"] == {"file_uri": "uri-x", "mime_type": "video/mp4"}
    assert parts[0]["video_metadata"] == {"fps": 1}
    assert parts[0]["media_processing"] == "STATIC"
    assert "text" in parts[1]
    cfg = req["config"]
    assert cfg["media_resolution"] == "MEDIA_RESOLUTION_LOW"
    assert cfg["thinking_config"] == {"thinking_level": "LOW"}
    assert cfg["response_mime_type"] == "application/json"
    assert "temperature" not in cfg
    assert req["model"] == "gemini-3.8-flash" and meta["kind"] == "single"


def test_paired_request_labels_slots_and_records_where_the_clip_is():
    req, meta = build_request(Job("c", "paired"), PAIRED, load_prompt("paired_v1"), SETTINGS, "clip", "rev")
    parts = req["contents"][0]["parts"]
    assert parts[0]["text"] == "Video A:" and parts[2]["text"] == "Video B:"
    a, b = parts[1]["file_data"]["file_uri"], parts[3]["file_data"]["file_uri"]
    assert {a, b} == {"clip", "rev"}
    assert (a == "clip") == (meta["clip_slot"] == "A")


def test_pair_order_is_reproducible_and_balanced():
    slots = [pair_order(Job(f"c{i}", "paired")) for i in range(400)]
    assert slots == [pair_order(Job(f"c{i}", "paired")) for i in range(400)]
    assert 160 < slots.count("A") < 240


def test_cache_key_changes_with_anything_that_changes_the_answer():
    p = load_prompt("default_v1")
    base = job_key(Job("c", "default"), COND, SINGLE, p, SETTINGS)
    assert base == job_key(Job("c", "default"), COND, SINGLE, p, SETTINGS)
    assert base != job_key(Job("c", "default", repeat=1), COND, SINGLE, p, SETTINGS)
    assert base != job_key(Job("c", "dense"), COND, DENSE, p, SETTINGS)
    assert base != job_key(Job("c", "default"), {**COND, "sha256": "c" * 64}, SINGLE, p, SETTINGS)
    assert base != job_key(Job("c", "default"), COND, SINGLE, p, {**SETTINGS, "model": "other"})


@pytest.mark.parametrize("verdict,score,flag,implaus", [("implausible", 1, True, 1.0), ("plausible", 5, False, 0.0),
                                                         ("plausible", 3, False, 0.5)])
def test_parse_single(verdict, score, flag, implaus):
    text = json.dumps({"reason": "r", "problem_type": "none", "plausibility_score": score, "verdict": verdict})
    out = parse_answer(text, {"kind": "single"})
    assert out["flag"] is flag and out["score"] == implaus


def test_parse_checklist_any_item():
    ans = {"reason": "r", "backward_motion": False, "unexplained_pause": True, "jump_or_teleport": False,
           "impossible_order": False, "plausibility_score": 4, "verdict": "plausible"}
    out = parse_answer(json.dumps(ans), {"kind": "checklist"})
    assert out["flag"] is False and out["any_check"] is True


@pytest.mark.parametrize("slot,choice,flag", [("A", "A", False), ("A", "B", True), ("B", "A", True),
                                              ("B", "B", False), ("A", "neither", True)])
def test_parse_paired(slot, choice, flag):
    text = json.dumps({"reason": "r", "plausible_video": choice, "confidence": 4})
    out = parse_answer(text, {"kind": "paired", "clip_slot": slot})
    assert out["flag"] is flag
    assert (out["score"] > 0.5) is flag


@pytest.mark.parametrize("text", [None, "", "not json", json.dumps({"verdict": "plausible"}),
                                  json.dumps({"reason": "r", "plausibility_score": 9, "verdict": "plausible"})])
def test_parse_errors(text):
    with pytest.raises(ParseError):
        parse_answer(text, {"kind": "single"})


def test_usage_and_cost():
    class U:
        prompt_token_count, candidates_token_count, thoughts_token_count = 1000, 100, 300
    t = usage_from(U())
    assert t == {"input_tokens": 1000, "output_tokens": 100, "thought_tokens": 300}
    assert usage_from({"promptTokenCount": 5, "candidatesTokenCount": 1}) == \
        {"input_tokens": 5, "output_tokens": 1, "thought_tokens": 0}
    pricing = {"input": 0.75, "output": 3.75, "batch_discount": 0.5}
    assert cost_usd(t, pricing, batch=False) == pytest.approx((1000 * 0.75 + 400 * 3.75) / 1e6)
    assert cost_usd(t, pricing, batch=True) == pytest.approx(cost_usd(t, pricing, batch=False) / 2)
