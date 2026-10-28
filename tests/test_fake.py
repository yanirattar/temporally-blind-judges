import json

from tbj.fake import _condition_from_uri, _Models
from tbj.judge import Job, build_request, load_prompt

SETTINGS = {"model": "x", "media_resolution": None, "thinking_level": None, "temperature": None}


def test_fake_reads_what_a_file_shows_not_its_name():
    # Byte-identical files share one upload, so the name can be another condition's.
    assert _condition_from_uri("fake://data/conditions/0001/reversal_strong__rev.mp4")[:2] == ("clean", "")
    assert _condition_from_uri("fake://data/conditions/0001/clean__rev.mp4")[:2] == ("reversal", "strong")
    assert _condition_from_uri("fake://data/conditions/0001/freeze_mild__rev.mp4")[:2] == ("freeze", "mild")


def test_fake_rates_follow_its_table():
    m = _Models(error_rate=0.0, seed=3)
    flags = {"clean": [], "reversal_strong": []}
    setup = {"kind": "single", "prompt": "default_v1", "fps": 1}
    for i in range(500):
        for cond in flags:
            req, _ = build_request(Job(f"{i}__{cond}", "default"), setup, load_prompt("default_v1"), SETTINGS,
                                   f"fake://data/conditions/{i:04d}/{cond}.mp4")
            flags[cond].append(json.loads(m.generate_content(**req).text)["verdict"] == "implausible")
    assert abs(sum(flags["clean"]) / 500 - 0.10) < 0.04
    assert abs(sum(flags["reversal_strong"]) / 500 - 0.40) < 0.06
