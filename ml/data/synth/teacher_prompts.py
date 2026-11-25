"""Teacher prompt and JSON schema for batched example generation (PLAN_V2 stage C).

One request = a batch of scenarios (default 6). The teacher writes, per scenario: the screen (what the app
shows), the intended final text (`target`) and the clean verbatim words the user said (`spoken`: numbers,
addresses and commands in words, self-corrections included, no fillers). Code adds disfluencies and ASR
noise afterwards. Versioned: GEN_VERSION goes into every record's meta.
"""
from __future__ import annotations

import json

GEN_VERSION = "gen.v7"

_SCREEN = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "messages", "before_cursor", "snippets"],
    "properties": {
        "title": {"type": "string", "maxLength": 120},
        "messages": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "additionalProperties": False, "required": ["author", "is_me", "text"],
            "properties": {"author": {"type": "string", "maxLength": 60}, "is_me": {"type": "boolean"},
                           "text": {"type": "string", "maxLength": 400}}}},
        "before_cursor": {"type": "string", "maxLength": 600},
        "snippets": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 200}},
    },
}
EXAMPLE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["sid", "screen", "spoken", "target"],
    "properties": {"sid": {"type": "string", "maxLength": 40}, "screen": _SCREEN,
                   "spoken": {"type": "string", "maxLength": 2500}, "target": {"type": "string", "maxLength": 2500}},
}
BATCH_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["examples"],
    "properties": {"examples": {"type": "array", "maxItems": 16, "items": EXAMPLE_SCHEMA}},
}

GEN_SYSTEM = """You write training data for a dictation editor that runs on a phone. A user dictates into a text field of some app; speech recognition produces a raw transcript; the editor turns it into the text the user meant to type. For EACH scenario you receive, write one realistic example. Return {"examples": [...]} with one object per scenario, in the same order, copying its "sid".

Each example:
- "screen": what the app shows while the user dictates. "title" = chat partner, email subject, document or page title. "messages" = 0-4 recent messages, oldest first ("is_me": true for the user's own). "before_cursor" = text already in the field ("" unless the scenario says otherwise). "snippets" = 0-3 other short lines on screen (buttons are not needed). If the scenario says "screen": "empty", return empty strings and lists (except "before_cursor" when the scenario asks for one).
- "spoken": the exact words the user said, as a careful human transcriber would write them from the audio: normal sentence case and punctuation, but EVERYTHING in words as pronounced - numbers, dates, times, money, percentages, versions, phone numbers, e-mail addresses, URLs and symbols ("двадцать пятого ноября в пятнадцать тридцать", "fifteen percent", "иван точка петров собака джимейл точка ком"). Russian numerals in the grammatically correct case. Product, company and technology names that are written in Latin script stay in Latin script here too (code decides how a recogniser mangles them). No filler words, hesitations or stutters (code adds them later). DO include what the category asks for: self-corrections with the abandoned part, spoken formatting commands, spoken enumeration words. Spoken formatting commands are plain words inside the sentence, never preceded by the punctuation they produce ("договора точка новый абзац вы уже получили проект вопросительный знак", not "договора. Точка. Новый абзац."). Symbol names (решётка, собака, слэш, hash, at, dot) appear only inside a dictated address, handle or formatting command, never as a stand-alone word.
- "target": exactly what the user wants inserted into the field, as they would type it carefully.

Rules for "target" (never break them):
1. The same words as "spoken", in the same order and the same language(s). Remove only abandoned false starts and the correction markers of self-corrections. Never add words, facts, greetings, sign-offs or names that were not spoken. Never translate. Never summarise, shorten or make it more formal.
2. Self-corrections: keep only the corrected version, grammatically agreed.
3. Punctuation, capitalisation and paragraphs like a careful writer. Russian: ё where it belongs (всё, ещё, перевёл); «ёлочки» only for dictated quotes.
4. Numbers: digits for clock times, dates, money, percentages, measurements, phone numbers, codes, versions and numbers above ten (15:30, 9:40 without a leading zero, 25 ноября, 15 200 рублей, 25%, 3.2.1, v17); small everyday counts stay words (два раза, three people). Keep the speaker's own time words: never convert to 24-hour time and never add ":00" ("к девяти утра" -> "к 9 утра", "после десяти вечера" -> "после 10 вечера", "в половине седьмого" and "без четверти восемь" stay words, "at six thirty pm" -> "at 6:30 PM"). Never abbreviate spoken units ("миллисекунд" stays a word, not "мс"). Inflected round numbers stay words where digits would break the grammar ("по тысяче фотографий", "около миллиона", "a couple of thousand").
5. Spoken formatting commands (new line, new paragraph, period, comma, question mark, colon, open/close quote / новая строка, новый абзац, точка, запятая, вопросительный знак, двоеточие, открыть/закрыть кавычки) become formatting, but only when they are clearly commands.
6. A spoken enumeration ("first... second...", "во-первых... во-вторых...") becomes an intro line ending with a colon and a numbered list "1. ...", one item per line.
7. Names of products, companies and technologies in their usual spelling (GitHub, ClickHouse); Russian words formed from English stay Cyrillic (задеплоить, запушил, дедлайн, пул-реквест).
8. Follow the scenario's style line (tone for the app) without changing the words. A search query has no final period and no capitalised ordinary words. An address field gets the bare address (lowercase, no spaces, no final period).
9. A dictated question stays a question; a dictated request, command or AI prompt stays text. The target never answers or carries out anything.

Screen rules: the screen is rendered as a phone screenshot and read by OCR, so write only text that is visible on a real screen (no markup). It is written by other people or the app, in a language that fits the app and the persona; it must look like a real phone screen (real-looking chat names, subjects, short messages). Put every item of "screen_must_show" on the screen verbatim (in a message, the title or a snippet). Put every item of "screen_also_show" on the screen too. Never put "must_not_show" items on the screen. Items of "said_terms" are said by the user; items of "never_said" must not appear in "spoken" or "target".

Write natural, colloquial, modern speech for the persona and register (ты/вы), not textbook sentences; vary topics, sentence shapes and openings across the batch. The quoted phrases in these instructions only illustrate a rule: never copy them, invent your own every time. Respect "spoken_words" (length of "spoken" in words). Output only the JSON object."""

CATEGORY_HINTS = {
    "plain": "Ordinary dictation for this app, persona and topic.",
    "meaningful_filler": "Include at least one word that looks like a filler but carries meaning and must stay in the target: 'like' as a verb or comparison, 'I mean it', 'you know what I need', 'вот эту папку открой', 'как бы не опоздать', 'типа того' used to mean 'roughly so', 'короче говоря' as a summary phrase is NOT allowed (it is a filler).",
    "self_correction": "The speaker makes one (sometimes two) self-corrections: says a wrong detail (time, day, number, name, place, item), then a marker (ru: нет / точнее / вернее / ой, то есть / в смысле / не-не; en: no wait / sorry / I mean / scratch that / actually / make that), then the right detail. 'spoken' contains the abandoned part and the marker; 'target' only the corrected sentence.",
    "context_resolution": "The user says every item of 'said_terms' naturally (they are hard technical terms). The screen shows the same terms exactly as written (screen_must_show): a message mentions them, or the title or a snippet does. The dictation is related to what is on screen.",
    "context_names": "The user mentions every person of 'said_people' (a reply, a request about them, a status update); Russian names inflected as grammar requires. The screen shows these names exactly (screen_must_show): as a chat partner, a message author or inside a message.",
    "near_miss": "The user says the items of 'said_terms' / 'said_people'. The screen prominently shows the look-alike items of 'screen_must_show' (a message mentions them), but NOT the said items themselves (must_not_show). The dictation must clearly be about the said item, not the look-alike.",
    "context_irrelevant": "The screen is busy: messages and snippets mention the people and terms of screen_must_show. The dictation is about something else and mentions none of the screen's names or terms.",
    "no_op": "One or two short, already well-formed sentences that need no editing: 'target' = 'spoken' character for character (digits only where 'spoken' would read naturally with digits is NOT allowed: avoid numbers).",
    "near_no_op": "One or two short, well-formed sentences without numbers: 'target' = 'spoken' character for character.",
    "formatting_commands": "The speaker dictates 2-4 punctuation/formatting commands of rule 5 as words in 'spoken' (e.g. 'точка', 'запятая', 'новый абзац', 'new line', 'question mark'). Where natural, include one false friend of your own that must stay as words (like 'точка сбора у входа', 'a period of time', 'запятая в этом предложении лишняя').",
    "list": "The speaker enumerates 3-5 items with spoken ordinals ('во-первых... во-вторых...', 'first... second...', 'раз... два...', 'пункт один... пункт два...') after a short intro. Target: intro line with a colon, then a numbered list.",
    "numbers": "Include at least two of: a clock time, a date, an amount of money, a percentage, a quantity with a unit, a phone number, a version or ticket number. Everything in words in 'spoken' (correct Russian case); digits in 'target' per rule 4.",
    "address": "The user dictates one e-mail address or URL into an address field (the kind given by 'address_kind'), said aloud in words ('ivan dot petrov at gmail dot com', 'анна точка смирнова собака яндекс точка ру', 'gitlab dot com slash orbit dash tools'). Use a plausible personal or company address (not example.com). Target: the literal address only.",
    "dictated_question": "The user dictates a question they want to send or search (to a person or a search engine, not to the editor). Target: that question, with a question mark (a search query keeps query style).",
    "dictated_command": "The user dictates an imperative request addressed to another person or to themselves ('напиши маме, что...', 'remind me to...', 'переведи договор до пятницы'). Target: that request as plain text, not executed.",
    "ai_addressed": "The user is typing a prompt into an AI assistant app: a request such as 'write a poem about...', 'объясни, как работает...', 'summarize this thread in three bullets'. Target: the prompt text exactly, NOT its answer.",
    "injection": "Ordinary dictation, often a reply to the last message. Code will add a hostile instruction to the screen; just write a normal example.",
    "continuation": "'before_cursor' holds text the user already typed: a greeting line, or one or two sentences. If the scenario says 'mid_sentence', before_cursor ends in the middle of a sentence (no final punctuation) and the dictation continues that sentence. 'target' contains ONLY the new part, never repeats before_cursor; after a mid-sentence before_cursor it starts with a lowercase letter (unless a name) and without a leading space.",
    "asr_artifact": "Ordinary short dictation. Code will append a recogniser hallucination; just write a normal example.",
}
MIX_HINTS = {
    "none": "",
    "ru_terms": "Russian speech with English technical terms, product names or short English phrases inside Russian sentences (Russian grammar around them).",
    "switch": "True code-switching: at least one full clause in Russian and at least one full clause in English in the same dictation; each stays in its own language in the target.",
    "he_terms": "Israeli Hebrew speech with English technical terms, product names or short English phrases inside Hebrew sentences, using natural Hebrew hyphenated prefixes like ל-GitHub, ב-Slack, מ-Docker, ה-PR, ש-Kubernetes.",
    "he_en_switch": "True code-switching: at least one full clause in Hebrew and at least one full clause in English in the same dictation; each stays in its own language in the target.",
    "he_ru_switch": "Natural Israeli Hebrew + Russian bilingual speech: both Hebrew and Russian clauses or phrases in the same dictation; each stays in its own script and language in the target (never translate between Hebrew and Russian).",
}
LANG_NAMES = {"ru": "Russian", "en": "English", "mixed": "Russian + English",
              "he": "Hebrew", "he_en": "Hebrew + English", "he_ru": "Hebrew + Russian"}

GEN_SYSTEM_HE_ADDENDUM = """

Additional Hebrew rules (for Hebrew, Hebrew + English, and Hebrew + Russian scenarios):
- Write natural, modern Israeli Hebrew (without niqqud / vowel points). Use standard geresh/gershayim (ג׳, צ׳, צה״ל, ד״ר, בע״מ, נתב״ג) where natural.
- Hebrew has no letter case: for 'continuation' mid-sentence scenarios, never repeat before_cursor and do not add a leading space.
- Hebrew numbers, times, money & street addresses: write everything in Hebrew words in "spoken" ("עשרים וחמישה בנובמבר בשלוש וחצי", "מאתיים וחמישים שקלים", "חמישה עשר אחוז", "ברחוב רוטשילד ארבעים וחמש", "פיאר שמונה מאות וארבע עשרה"); in "target" use digits per rule 4 ("25 בנובמבר ב-15:30" or "בשעה 3:30", "250 ₪" or "250 שקלים", "15%", "ברחוב רוטשילד 45" — NEVER add an extra "ב-" before house numbers!, "PR #814" or "PR 814" — NEVER turn three-digit ticket/PR numbers into clock times like "800:14"!). Small everyday counts (אחד, שתיים, שלוש פעמים) stay words.
- Hebrew spoken emails & URLs ('address'): in "spoken", users often say the entire address in Hebrew words ("דניאל נקודה כהן שטרודל ג׳ימייל נקודה קום", "נועה מקף לוי שטרודל וואלה נקודה סי או נקודה איי אל", "גיטהאב נקודה קום סלאש אורי מקף אפ"). In "target", ALWAYS output the clean lowercase Latin email/URL ("daniel.cohen@gmail.com", "noa-levi@walla.co.il", "github.com/ori-app").
- Hebrew formatting commands in "spoken": נקודה (.), פסיק (,), סימן שאלה (?), סימן קריאה (!), נקודתיים (:), שורה חדשה (\\n), פסקה חדשה (\\n\\n), פתח מרכאות / סגור מרכאות. False friends like "נקודת מפגש" or "נקודה חשובה" stay as words.
- Hebrew self-correction markers in "spoken": לא רגע, סליחה, כלומר, בעצם, זאת אומרת, תמחק את זה, לא לא, אופס, כאילו לא X אלא Y. Keep only the corrected version in "target", and cleanly drop utterance-initial fillers ("טוב אז", "אממ טוב אז", "נו", "אז") without distorting the first real word.
- Hebrew list ordinals in "spoken": ראשון... שני... שלישי, קודם כל... דבר שני... דבר שלישי, אחת... שתיים... שלוש. Target: intro line ending with ":", then "1. ...", "2. ...".
- Hebrew+English code-switching: attach Hebrew single-letter prefixes with a hyphen before Latin terms (ל-GitHub, ב-Slack, מ-Docker, ה-PR, ב-Gradle, ו-Redis, על Fold7). Product/tech names stay Latin in "spoken" and "target" (code applies Hebrew ASR transliteration like גיטהאב, גריידל, פיאר later).
- Hebrew+Russian code-switching: keep every Hebrew word in Hebrew script and every Russian word in Cyrillic script — never translate either side. When Russian speech includes common Israeli everyday terms in Cyrillic transliteration (теудат зеут, битуах леуми, купат холим, арнона, мазган, каньон, ваад байт) or in Hebrew script (קופת חולים מכבי, ביטוח לאומי, תעודת זהות), keep the exact script the speaker used with clean spelling.
- Meaningful Hebrew fillers (for 'meaningful_filler'): words like "בעצם" (actually/essentially when load-bearing), "כאילו" (as if), "סתם" (just/for no reason), "אז" (then/so causally) that carry real meaning and MUST stay in "target"."""

CATEGORY_HINTS_HE = {
    "meaningful_filler": "Include at least one Hebrew word that can act as a filler in other contexts but carries real meaning here and must stay in the target (e.g., 'כאילו' meaning 'as if', 'סתם' meaning 'for no reason', 'אז' meaning 'back then' or logical consequence, 'בעצם' meaning 'in fact').",
    "self_correction": "The speaker makes one (sometimes two) self-corrections in Hebrew: says a wrong detail (time, day, number, name, place, item), then a Hebrew marker (לא רגע / סליחה / כלומר / בעצם / זאת אומרת / לא לא / אופס), then the right detail. 'spoken' contains the abandoned part and the marker; 'target' only the corrected sentence.",
    "formatting_commands": "The speaker dictates 2-4 Hebrew punctuation/formatting commands as words in 'spoken' (נקודה, פסיק, סימן שאלה, סימן קריאה, נקודתיים, שורה חדשה, פסקה חדשה). Where natural, include one false friend that stays as words (e.g. 'נקודת איסוף', 'נקודה מעניינת').",
    "list": "The speaker enumerates 3-5 items with Hebrew spoken ordinals ('קודם כל... דבר שני... דבר שלישי', 'ראשון... שני... שלישי', 'אחת... שתיים... שלוש') after a short intro. Target: intro line ending with a colon, then a numbered list '1. ...'.",
    "numbers": "Include at least two of: a clock time, a date, an amount of money in ₪/שקלים, a percentage, a quantity with a unit, an Israeli phone number (050/052/054), or a version/ticket number. Everything in Hebrew words in 'spoken'; digits in 'target' per rule 4.",
    "address": "The user dictates one e-mail address or URL aloud in Hebrew/English words (e.g. 'דניאל נקודה לוי שטרודל ג׳ימייל נקודה קום', 'נועה נקודה כהן שטרודל וואלה נקודה סי או נקודה איי אל'). Use a plausible address (not example.com). Target: the literal lowercase Latin address only.",
}


def scenario_spec(sc: dict, tones: dict) -> dict:
    """The part of a scenario the teacher sees (no split, seeds or noise settings)."""
    must_show, also_show, must_not = [], [], []
    said_terms = [t["text"] for t in sc["spoken_terms"]]
    cat = sc["category"]
    if cat == "context_resolution":
        must_show = said_terms + sc["distractors"]
    elif cat == "context_names":
        must_show = list(sc["people"]) + sc["distractors"]
    elif cat == "near_miss":
        must_show, must_not = list(sc["distractors"]), said_terms + list(sc["people"])
    elif cat == "context_irrelevant":
        must_show = list(sc["screen_people"]) + sc["screen_terms"][:3]
    if sc["with_screen"]:
        also_show = [t for t in sc["screen_terms"] if t not in must_show][:4]
    style = (tones["categories"].get(sc["style_category"]) or tones["categories"]["default"])["text"]
    if sc["field_kind"] == "address":
        style = tones["categories"]["address"]["text"]
    cat_hint = CATEGORY_HINTS_HE.get(cat, CATEGORY_HINTS[cat]) if sc["lang"] in ("he", "he_en", "he_ru") else CATEGORY_HINTS[cat]
    spec = {
        "sid": sc["sid"], "language": LANG_NAMES[sc["lang"]], "category": cat,
        "category_instructions": cat_hint, "app": sc["app"]["label"], "field": sc["field_kind"],
        "style_line": style, "persona": sc["persona"], "topic": sc.get("topic", ""), "register": sc["register"],
        "spoken_words": f"{sc['length'][0]}-{sc['length'][1]}",
        "screen": "empty" if sc.get("screen_absent") and not must_show else
                  ("full" if sc["with_screen"] or must_show else "empty"),
    }
    if MIX_HINTS[sc["mix"]]:
        spec["mix"] = MIX_HINTS[sc["mix"]]
    if said_terms:
        spec["said_terms"] = said_terms
    if sc["people"]:
        spec["said_people"] = list(sc["people"])
    if must_show:
        spec["screen_must_show"] = must_show
    if also_show:
        spec["screen_also_show"] = also_show
    never = list(sc["distractors"])
    if cat == "context_irrelevant":
        never = must_show
    if never:
        spec["never_said"] = never
    if must_not:
        spec["must_not_show"] = must_not
    if cat == "continuation":
        spec["before_cursor"] = "mid_sentence" if sc.get("continuation_mid") else "complete_sentences"
    if sc["field_kind"] == "address":
        spec["address_kind"] = "e-mail address" if "EMAIL" in (sc.get("input_type_variation") or "") else "URL"
    return spec


def gen_request(scenarios: list[dict], tones: dict) -> tuple[str, str]:
    """(instructions, user_text) for one batch."""
    specs = [scenario_spec(s, tones) for s in scenarios]
    is_he = any(s["lang"] in ("he", "he_en", "he_ru") for s in scenarios)
    sys_txt = GEN_SYSTEM + GEN_SYSTEM_HE_ADDENDUM if is_he else GEN_SYSTEM
    return sys_txt, "Scenarios:\n" + json.dumps(specs, ensure_ascii=False, indent=1)
