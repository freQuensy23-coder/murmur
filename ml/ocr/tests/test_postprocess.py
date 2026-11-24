from ml.ocr import postprocess as pp
from ml.ocr.config import DEFAULT


def L(text, box, conf=0.9):
    return {"text": text, "conf": conf, "box": list(box)}


def test_homoglyph_mixed_tokens():
    assert pp.repair_homoglyphs("kлactepy") == "кластеру"
    assert pp.repair_homoglyphs("Grafanа") == "Grafana"
    assert pp.repair_homoglyphs("Привет, Grafana!") == "Привет, Grafana!"


def test_homoglyph_all_twin_tokens_follow_unanimous_neighbours():
    # "Bce" recognised in Latin between Russian words -> Cyrillic
    assert pp.repair_homoglyphs("Bce готово к релизу") == "Все готово к релизу"
    # Cyrillic "с" between a Russian and an English word stays as recognised
    assert pp.repair_homoglyphs("Встреча с Ingrid") == "Встреча с Ingrid"
    assert pp.repair_homoglyphs("Встреча c Ingrid") == "Встреча c Ingrid"
    # mixed token in Latin context -> Latin; a twin-only Russian word between English terms stays
    assert pp.repair_homoglyphs("open the АРI docs") == "open the API docs"
    assert pp.repair_homoglyphs("Grafana по kube-prometheus") == "Grafana по kube-prometheus"


def test_homoglyph_leaves_digits_and_symbols():
    assert pp.repair_homoglyphs("PR #4182 15:30") == "PR #4182 15:30"
    assert pp.repair_homoglyphs("") == ""


def test_filter():
    assert pp.keep(L("ok", (0, 0, 1, 1), 0.9), DEFAULT)
    assert not pp.keep(L("a", (0, 0, 1, 1), 0.9), DEFAULT)
    assert not pp.keep(L("hello", (0, 0, 1, 1), 0.3), DEFAULT)


def test_exclusion():
    assert pp.excluded([0, 0, 10, 10], [(0, 0, 10, 6)], 0.5)
    assert not pp.excluded([0, 0, 10, 10], [(0, 0, 10, 4)], 0.5)
    assert not pp.excluded([0, 0, 10, 10], [], 0.5)


def test_dedup():
    lines = [L("abc", (0, 0, 100, 20), 0.8), L("abc", (2, 1, 100, 21), 0.9), L("x y", (0, 50, 50, 70))]
    g = pp.dedup_geometric(lines, 0.5)
    assert [ln["conf"] for ln in g] == [0.9, 0.9]
    t = pp.dedup_text([L("12:41", (0, 0, 5, 5)), L("Hi", (0, 10, 5, 15)), L("12:41", (0, 20, 5, 25))])
    assert [ln["text"] for ln in t] == ["12:41", "Hi"]


def test_rows_left_to_right():
    lines = [L("time", (800, 104, 900, 130)), L("text", (50, 100, 600, 140)), L("next", (50, 160, 300, 190))]
    out = pp.reading_order(lines, 1080, 2520, DEFAULT)
    assert [ln["text"] for ln in out] == ["text", "time", "next"]


def test_two_pane():
    W, H = 1968, 2184
    head = [L("header", (40, 120, 1900, 180))]
    left = [L(f"L{i}", (40, 300 + i * 200, 700, 360 + i * 200)) for i in range(6)]
    right = [L(f"R{i}", (900, 280 + i * 200, 1900, 340 + i * 200)) for i in range(6)]
    out = pp.reading_order(head + right + left, W, H, DEFAULT)
    assert [ln["text"] for ln in out] == ["header"] + [f"L{i}" for i in range(6)] + [f"R{i}" for i in range(6)]


def test_no_two_pane_on_cover_or_crossing_lines():
    lines = [L(f"L{i}", (40, 300 + i * 200, 500, 360 + i * 200)) for i in range(6)] + \
            [L(f"R{i}", (600, 400 + i * 200, 1040, 460 + i * 200)) for i in range(6)]
    out = pp.reading_order(lines, 1080, 2520, DEFAULT)   # narrow display: plain rows
    assert [ln["text"] for ln in out][:3] == ["L0", "R0", "L1"]
    W, H = 1968, 2184
    lines2 = lines + [L("wide", (40, 900, 1500, 950))]
    assert pp.find_gutter(lines, W, H, DEFAULT) is not None
    assert pp.find_gutter(lines2, W, H, DEFAULT) is None   # a line crossing the gutter inside the band


def test_screen_level_default_script():
    assert pp.screen_script(["Привет, как дела", "Мама"], 2.0) == "cyr"
    assert pp.repair_homoglyphs("Bce", "cyr") == "Все"
    assert pp.repair_homoglyphs("Mama", "cyr") == "Мата"     # known miss: Latin m is т or м, т is twice as frequent
    assert pp.repair_homoglyphs("CEO", "cyr") == "CEO"        # acronyms stay
    assert pp.repair_homoglyphs("Mama", None) == "Mama"
    assert pp.repair_homoglyphs("Меню SІМ-карты") == "Меню SIM-карты"


def test_twin_only_rules():
    r = pp.repair_homoglyphs
    assert r("Посмотри НАРгоху, пожалуйста") == "Посмотри HAProxy, пожалуйста"     # camel case
    assert r("Обнови АРІ до вечера") == "Обнови API до вечера"                    # non-Russian І
    assert r("сервер А100 и Н100, х86_64, р99") == "сервер A100 и H100, x86_64, p99"   # identifiers
    assert r("Посмотри promрt injection") == "Посмотри prompt injection"          # majority of letters
    assert r("Google Мeet") == "Google Meet"
    assert r("LӀamaIndex") == "LlamaIndex"
    assert r("Прiоритет") == "Приоритет"
    assert r("доступ к ClickHouse и Kafka") == "доступ к ClickHouse и Kafka"


def test_glued_subtokens():
    assert pp.repair_homoglyphs("Обнови RNN-Т до вечера") == "Обнови RNN-T до вечера"
    assert pp.repair_homoglyphs("Llama.срр") == "Llama.cpp"
    assert pp.repair_homoglyphs("Меню SIM-карты") == "Меню SIM-карты"


def test_visual_to_logical_hebrew_and_mixed():
    v2l = pp.visual_to_logical
    # Pure Hebrew (visual LTR -> logical RTL)
    assert v2l("םלוע םולש") == "שלום עולם"
    assert v2l("!םלוע םולש") == "שלום עולם!"
    assert v2l("(תירבע) םולש") == "שלום (עברית)"
    assert v2l("ל״הצ") == "צה״ל"
    assert v2l("םולש ר״ד") == "ד״ר שלום"
    # Mixed he+en (RTL-dominant, preserving LTR runs intact)
    assert v2l("GitHub-ל push יתישע") == "עשיתי push ל-GitHub"
    assert v2l(".GitHub-ל push יתישע") == "עשיתי push ל-GitHub."
    assert v2l("kube-prometheus רובע Grafana קניל") == "לינק Grafana עבור kube-prometheus"
    assert v2l("Hello World םולש") == "שלום Hello World"
    # Mixed he+digits
    assert v2l("15:30-ב השגפ") == "פגשה ב-15:30"
    assert v2l("50₪ ריחמ") == "מחיר 50₪"
    assert v2l("₪50 ריחמ") == "מחיר ₪50"
    assert v2l("₪ 50 ריחמ") == "מחיר 50 ₪"
    assert v2l("100% חוטב") == "בטוח 100%"
    assert v2l("054-123-4567 רפסמ") == "מספר 054-123-4567"
    assert v2l("באהטיגב PR #4182 יתחתפ") == "פתחתי PR #4182 בגיטהאב"
    # LTR-dominant lines with embedded Hebrew
    assert v2l("Hello םולש world") == "Hello שלום world"
    assert v2l("Hello םלוע םולש world") == "Hello שלום עולם world"
    assert v2l("Привет םולש мир") == "Привет שלום мир"
    # Non-Hebrew RU/EN 100% untouched
    assert v2l("Привет, Grafana! 15:30") == "Привет, Grafana! 15:30"
    assert v2l("PR #4182 (main) — 100%") == "PR #4182 (main) — 100%"
    assert v2l("") == ""


def test_rows_rtl_right_to_left_and_ltr_unchanged():
    # RTL-dominant row sorts right-to-left (rightmost box first)
    heb_row = [
        L("12:30", (50, 104, 150, 130)),
        L("שלום עולם", (500, 100, 980, 140)),
        L("עוד שורה", (500, 160, 980, 195)),
    ]
    out_heb = pp.reading_order(heb_row, 1080, 2520, DEFAULT)
    assert [ln["text"] for ln in out_heb] == ["שלום עולם", "12:30", "עוד שורה"]
    # Two Hebrew boxes on the same row: right box before left box
    two_heb = [L("שמאל", (100, 100, 400, 140)), L("ימין", (600, 102, 950, 142))]
    assert [ln["text"] for ln in pp.reading_order(two_heb, 1080, 2520, DEFAULT)] == ["ימין", "שמאל"]
    # LTR row remains left-to-right
    ltr_row = [L("12:30", (900, 104, 1000, 130)), L("Привет мир", (50, 100, 600, 140))]
    assert [ln["text"] for ln in pp.reading_order(ltr_row, 1080, 2520, DEFAULT)] == ["Привет мир", "12:30"]
