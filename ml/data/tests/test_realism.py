"""Realism guards added after the 2026-11-19 review: ASR noise plausibility, code-switch segmentation,
per-model disfluency rendering, recoverability, review flags, eval contamination, app/screen coherence."""
import random
from pathlib import Path

import pytest

from ml.common.render_bridge import REPO
from ml.data import contam as C
from ml.data.synth import build as B
from ml.data.synth import disfluency as D
from ml.data.synth import filters as F
from ml.data.synth import sampler as S
from ml.data.synth import screen as SCR
from ml.data.synth.channel import (Channel, _lang_runs, _letter_drop, _term_garble_implausible, plausible_ins,
                                   plausible_sub)

MINED = REPO / "ml" / "roundtrip" / "samples" / "mined" / "real"
needs_mined = pytest.mark.skipif(not (MINED / "profiles.json").exists(), reason="mined tables absent")


def test_plausible_substitutions_and_insertions():
    for a, b in [("the", "that"), ("a", "the"), ("конца", "коны"), ("Скинь", "В скин"), ("Я", "Е"), ("for", "from")]:
        assert plausible_sub(a, b), (a, b)
    for a, b in [("по", "девять"), ("the", "but"), ("стенде", "съем"), ("шесть", "съем"), ("I’ll", "And"),
                 ("вы", "ини"), ("пожалуйста", "найденный"), ("вопросительный", "пользователей")]:
        assert not plausible_sub(a, b), (a, b)
    assert all(plausible_ins(w) for w in ["и", "ну", "the", "uh", "э"])
    assert not any(plausible_ins(w) for w in ["пост", "папы", "девяносто", "радел", "coup"])


def test_letter_drop_kinds():
    assert _letter_drop("Kubernetes", "Kuberetes") == "interior"
    assert _letter_drop("GitLab", "Gtlab") == "interior"
    assert _letter_drop("Ollama", "Olama") == "phonetic"
    assert _letter_drop("htop", "top") == "phonetic"
    assert _letter_drop("Docker", "докер") is None
    none = {"err": 50, "cyr_share": 0.9, "drop_acronym": 0, "drop_other": 0, "drop_interior": 0}
    seen = {"err": 50, "cyr_share": 0.1, "drop_acronym": 3, "drop_other": 4, "drop_interior": 3}
    assert _term_garble_implausible("Kubernetes", "Kuberetes", none)
    assert not _term_garble_implausible("Kubernetes", "Kuberetes", seen)
    assert _term_garble_implausible("Pro", "п", seen) and _term_garble_implausible("PR", "P", seen)
    assert not _term_garble_implausible("LLM", "LM", seen) and _term_garble_implausible("LLM", "LM", none)


def test_language_runs():
    segs = _lang_runs("Я закинул фикс в main, but the build is still red, глянь пожалуйста.")
    assert [k for _, k in segs] == ["ru", "en", "ru"]
    assert _lang_runs("Задеплой на staging сегодня.") == [("Задеплой на staging сегодня.", "ru")]


@needs_mined
def test_switch_clause_is_written_in_english_by_latin_models():
    ch = Channel(MINED)
    m = next(x for x in ch.models("mixed") if "whisper-large" in x)
    assert ch.can_switch(m) and not ch.can_switch(next(x for x in ch.models("mixed") if "t-one" in x))
    text = "Я посмотрю вечером, but I think the release is ready for tomorrow, напишу тебе потом."
    for seed in range(20):
        hyp, ev = ch.transcribe(text, text, "mixed", m, [], seed, 1.0)
        assert any(e.startswith("segments:") for e in ev)
        assert not any(e.startswith("term:") and ":garbled" in e for e in ev), ev  # ordinary words are not terms
        assert "think" in hyp.lower() or "release" in hyp.lower()


@needs_mined
def test_implausible_term_garbles_are_repaired_for_cyrillic_models():
    ch = Channel(MINED)
    m = next(x for x in ch.models("ru") if "t-one" in x)
    bad = ch.implausible(["term:Kubernetes->Kuberetes:garbled", "sub:на->а"], m, "ru")
    assert bad == ["term:Kubernetes->Kuberetes:garbled"]
    hyp, ev, rest = ch.repair_terms("подними kuberetes на стенде", ["term:Kubernetes->Kuberetes:garbled"], m, "ru",
                                    bad, [{"text": "Kubernetes", "ru_spoken": ["кубернетис"]}])
    assert rest == [] and hyp == "подними кубернетис на стенде" and ev[0].endswith(":translit_repair")


@needs_mined
def test_disfluency_render_per_model():
    ch = Channel(MINED)
    wl = ch.disfluency_render(next(x for x in ch.models("ru") if "whisper-large" in x))
    assert wl["hesitation_keep"] <= 0.3
    r = random.Random(3)
    text = "Я сегодня приду позже, давай созвонимся вечером, если получится, и всё обсудим подробно."
    for seed in range(30):
        out, ops = D.inject(text, "ru", 0.8, random.Random(seed), sounds_only=True,
                            render={"hesitation_keep": 0.0, "repeat_keep": 0.0})
        assert D.hesitation_tokens(out) == 0
        assert all(o.get("suppressed") for o in ops if o["op"] in ("hesitation", "repeat", "restart"))
    assert "слушай" not in D.FILLERS["ru"] + D.START_FILLERS["ru"]
    del r


def test_recoverability_needs_real_evidence():
    assert B.unrecoverable("Вот этот MacBook Pro забери.", "Вот этот пост Macbok п забери", "") == ["pro"]
    assert B.unrecoverable("Вот этот MacBook Pro забери.", "Вот этот пост Macbok п забери", "МасВоок Pro") == []
    assert B.unrecoverable("Начнём в 6 вечера.", "начнём в съем вечера", "") == ["6"]
    assert B.unrecoverable("Начнём в 6 вечера.", "начнём в шесть вечера", "") == []
    assert B.unrecoverable("Docker упал.", "докер упал", "") == []


def test_review_flags():
    rec = {"category": "plain", "ctx": {"v": 2, "app": {"category": "messaging"}}, "transcript": "and And send you an ETA",
           "target": "and I’ll send you an ETA tomorrow", "meta": {"spoken": "and I’ll send you an ETA"}}
    assert F.review_flags(rec) == ["tomorrow"]


def test_teacher_prompt_and_eval_contamination():
    assert C.prompt_hits() == []
    eid, _f, text = next(t for t in C.eval_texts() if t[1] == "target" and len(t[2]) > 30)
    assert C.contaminated({"target": text, "transcript": "x"}) is not None
    assert C.contaminated({"target": "Совсем другой текст про погоду и прогулку в парке.", "transcript": ""}) is None


def test_apps_categories_and_templates_are_coherent():
    from ml.screens.templates import TEMPLATE_APPS  # noqa: F401  (skips cleanly below if screens are absent)
    plan = S.plan(300, 30, 30, seed=7)
    for sc in plan:
        app = sc["app"]
        assert app["category"] == S.app_category(app["package"]), app
        assert app["package"] != "com.example.app"
        tpl = sc["render"]["screen_template"]
        fam = TEMPLATE_APPS[tpl][4]
        if app["category"] == "messaging":
            assert fam in ("messenger", "work_chat"), (app, tpl)
        if app["category"] == "work_chat":
            assert tpl in ("work_chat", "issue_tracker"), (app, tpl)
        if sc["field_kind"] == "address":
            assert app["category"] in ("email", "search"), app
        if app["package"] == "com.termux":
            pytest.fail("prose dictation into a terminal")


def test_copied_prompt_examples_are_rejected():
    assert F._copies_prompt_example("Новый абзац. Точка сбора у входа во двор.")
    assert not F._copies_prompt_example("Встречаемся у входа во двор в семь.")
