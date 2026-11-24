"""Unit tests for RTL Hebrew screen rendering in ml/screens/."""
from __future__ import annotations

import random

import pytest

from ml.screens import chrome as C
from ml.screens.filler import UI
from ml.screens.render import build_html
from ml.screens.specs import HINTS, from_teacher
from ml.screens.templates import TEMPLATES, normalise_spec

HE_SCREEN = {
    "title": "נועה כהן",
    "messages": [
        {"author": "נועה כהן", "is_me": False, "text": "היי, בדקת את ה-PR ב-GitHub?"},
        {"author": "", "is_me": True, "text": "כן, הכל נראה טוב, מעלה ל-staging היום."},
    ],
    "before_cursor": "אני שולח לך את",
    "snippets": ["עדכון גרסה ב-Kubernetes"],
}


def test_hebrew_ui_and_hints_complete():
    # Every UI key in English/Russian exists in Hebrew
    assert set(UI["en"]).issubset(set(UI["he"]))
    assert UI["he"]["send"] == "שלח"
    assert UI["he"]["search"] == "חיפוש"
    assert UI["he"]["today"] == "היום"
    assert set(HINTS["en"]) == set(HINTS["he"])


def test_every_template_builds_rtl_html_for_hebrew():
    for t in TEMPLATES:
        for dev in ("fold7_cover", "fold7_inner"):
            spec = from_teacher(
                HE_SCREEN,
                {"package": "com.whatsapp", "label": "WhatsApp", "category": "messaging"},
                "message",
                "he",
                seed=7,
                template=t,
                device=dev,
                locale="he",
                keyboard=True,
            )
            html, d = build_html(spec)
            assert '<html dir="rtl" lang="he">' in html, t
            assert "Noto Sans Hebrew" in html and "Rubik" in html and "Heebo" in html and "Arial" in html
            assert "ocr_fonts/NotoSansHebrew.ttf" in html
            assert 'data-exclude="keyboard"' in html
            assert d["w"] in (1080, 1968)


def test_hebrew_keyboard_and_bubble():
    kb = C.keyboard("he", random.Random(1), numbers=True)
    assert "עברית" in kb
    assert "ק" in kb and "ש" in kb and "ץ" in kb
    # Bubble pill with Hebrew locale
    pills = [C.murmur_bubble(random.Random(i), 411, 959, 90, 400, "he") for i in range(30)]
    assert any("מקשיב…" in p for p in pills)


def test_from_teacher_hebrew_langs_and_ltr_untouched():
    for lang in ("he", "he_en", "he_ru"):
        locs = {
            from_teacher(HE_SCREEN, {"package": "org.telegram.messenger", "label": "Telegram", "category": "messaging"},
                         "message", lang, seed=s)["locale"]
            for s in range(20)
        }
        assert "he" in locs

    ru_spec = from_teacher(
        {"title": "Анна", "messages": [{"author": "Анна", "is_me": False, "text": "Привет!"}],
         "before_cursor": "", "snippets": []},
        {"package": "org.telegram.messenger", "label": "Telegram", "category": "messaging"},
        "message",
        "ru",
        seed=2,
        template="messenger_a",
        locale="ru",
    )
    ru_html, _ = build_html(ru_spec)
    assert "<!doctype html><html>" in ru_html
    assert 'dir="rtl"' not in ru_html.split("</style>", 1)[1]


def test_hebrew_message_gets_rtl_even_under_en_locale():
    spec = from_teacher(
        HE_SCREEN,
        {"package": "com.whatsapp", "label": "WhatsApp", "category": "messaging"},
        "message",
        "he_en",
        seed=1,
        template="messenger_b",
        locale="en",
    )
    html, _ = build_html(spec)
    assert 'dir="rtl"' in html


def test_invalid_locale_rejected():
    with pytest.raises(ValueError):
        normalise_spec({"template": "notes", "locale": "xx"})
