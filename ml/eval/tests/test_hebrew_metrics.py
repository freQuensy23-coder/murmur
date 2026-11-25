"""Unit tests for Hebrew (he, he_en, he_ru) evaluation contract and metrics."""
from __future__ import annotations

from ml.eval import contract, gates, metrics


def _ex(target, transcript="x", category="plain", lang="he", ctx=None, check=None, **meta):
    m = dict(meta)
    if check:
        m["check"] = check
    return {"id": "he-1", "lang": lang, "category": category, "source": "sim", "split": "test",
            "ctx": ctx or {}, "transcript": transcript, "target": target, "meta": m}


def test_contract_accepts_hebrew_langs():
    for lang in ("he", "he_en", "he_ru"):
        contract.validate(_ex("שלום עולם.", lang=lang))


def test_wrong_language_catches_hebrew_translations():
    he_ex = _ex("שלום, בוא ניפגש מחר בבוקר במשרד.", lang="he")
    assert not metrics.score_example(he_ex, "שלום, בוא ניפגש מחר בבוקר במשרד")["wrong_lang"]
    # Translated to English -> wrong_lang
    assert metrics.score_example(he_ex, "Hello, let's meet tomorrow morning at the office.")["wrong_lang"]
    # Translated to Russian -> wrong_lang
    assert metrics.score_example(he_ex, "Привет, давай встретимся завтра утром в офисе.")["wrong_lang"]

    # Mixed he_en keeps both scripts
    he_en = _ex("עשיתי push ל-GitHub ופתחתי pull request.", lang="he_en")
    assert not metrics.score_example(he_en, "עשיתי push ל-GitHub ופתחתי pull request.")["wrong_lang"]
    assert metrics.score_example(he_en, "I pushed to GitHub and opened a pull request.")["wrong_lang"]

    # Mixed he_ru keeps both scripts
    he_ru = _ex("יאללה, בוא נדבר מחר, я скину детали.", lang="he_ru")
    assert not metrics.score_example(he_ru, "יאללה, בוא נדבר מחר, я скину детали.")["wrong_lang"]
    assert metrics.score_example(he_ru, "Давай поговорим завтра, я скину детали.")["wrong_lang"]


def test_hebrew_negation_drift_and_preamble_detection():
    e = _ex("אני לא מגיע היום לפגישה ב-15:30.", lang="he")
    assert metrics.score_example(e, "אני מגיע היום לפגישה ב-15:30.")["drift"]
    assert not metrics.score_example(e, "אני לא מגיע היום לפגישה ב-15:30")["drift"]

    q = _ex("מתי נפתח המשרד מחר?", "מתי נפתח המשרד מחר", category="dictated_question", lang="he")
    assert metrics.score_example(q, "בטח, הנה הטקסט: מתי נפתח המשרד מחר?")["answered"]
    assert not metrics.score_example(q, "מתי נפתח המשרד מחר?")["answered"]


def test_summarize_and_gates_include_hebrew_scopes():
    rows = []
    for i, (lg, txt) in enumerate([("ru", "Привет."), ("he", "שלום."), ("he_en", "בדיקה ב-GitHub.")]):
        e = _ex(txt, txt, category="no_op", lang=lg)
        e["id"] = f"r-{i}"
        rows.append(metrics.score_example(e, txt))
    s = metrics.summarize(rows)
    assert set(s["by_lang"]) == {"ru", "he", "he_en"}
    g = gates.evaluate(s)
    assert {"overall", "ru", "he", "he_en"} == set(g["scopes"])
