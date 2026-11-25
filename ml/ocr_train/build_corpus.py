"""Build a balanced 3-language (ru + en + he) text corpus with visual-order CTC labels.

Target mixture (configurable):
  - 40% he    : Hebrew (HF wikimedia/wikipedia he + ivrit-ai transcripts + Israeli names/services/prices/UI)
  - 15% he_en : Hebrew + English code-switching (ל-GitHub, ב-Slack, ה-PR, tech/chat)
  - 35% ru    : Russian + Russian-English (split ~22% ru, ~13% ru_en; HF wikipedia ru + messenger/email/tech)
  -  5% en    : English / code / URLs / CLI / numbers (HF wikipedia en + dev/UI)
  -  5% ru_he : Russian + Hebrew bilingual lines (Привет שלום, Olim chat, Israeli services)

Outputs:
  <out_dir>/train_corpus.jsonl
  <out_dir>/val_corpus.jsonl
  <out_dir>/corpus_stats.json

Each JSONL record:
  {"id": int, "text": str, "visual": str, "base_dir": "L"|"R", "cat": str}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .charset import (
    DEFAULT_DICT_PATH,
    Charset,
    detect_base_dir,
    has_hebrew,
    load_charset,
    strip_nikkud,
    to_visual_order,
)

# ----------------------------------------------------------------------------- vocabulary & templates

HE_FIRST_NAMES = [
    "דניאל", "נועה", "איתי", "תמר", "עומר", "שירה", "אורי", "מאיה", "יונתן", "יעל",
    "רועי", "אביגיל", "גיא", "מיכל", "אלון", "עדי", "אריאל", "ליאור", "נדב", "רותם",
    "אייל", "הילה", "תומר", "דנה", "אסף", "קרן", "יובל", "גל", "עמית", "טל",
    "משה", "דוד", "אברהם", "שרה", "רחל", "לאה", "יוסף", "חיים", "אסתר", "מרים",
    "אלכס", "מקס", "אנה", "קטיה", "בוריס", "סרגיי", "נטשה", "איליה", "ולדימיר", "אולגה",
]

HE_LAST_NAMES = [
    "כהן", "לוי", "מזרחי", "פרץ", "ביטון", "דהן", "אברהם", "פרידמן", "אזולאי", "מלכה",
    "שפירא", "גולדשטיין", "כץ", "קליין", "רוזנברג", "שמש", "חדד", "גבאי", "אוחיון", "בן דוד",
    "אלמליח", "אמסלם", "אלבז", "סויסה", "וקנין", "נחום", "חיים", "יוסף", "שרון", "ברק",
]

HE_CITIES = [
    "תל אביב", "תל־אביב", "ירושלים", "חיפה", "ראשון לציון", "פתח תקווה", "אשדוד", "נתניה",
    "באר שבע", "חולון", "בני ברק", "רמת גן", "רחובות", "בת ים", "אשקלון", "הרצליה",
    "כפר סבא", "רעננה", "מודיעין", "גבעתיים", "נהריה", "אילת", "עכו", "טבריה", "צפת",
    "הוד השרון", "נס ציונה", "קריית אונו", "ראש העין", "יבנה",
]

HE_STREETS = [
    "רח׳ דיזנגוף", "רח׳ רוטשילד", "רח׳ אלנבי", "רח׳ אבן גבירול", "רח׳ בן יהודה",
    "רח׳ הרצל", "שד׳ נורדאו", "דרך בגין", "רח׳ קינג ג׳ורג׳", "רח׳ יפו", "רח׳ ויצמן",
    "רח׳ ז׳בוטינסקי", "רח׳ ביאליק", "רח׳ סוקולוב", "דרך נמיר", "רח׳ הארבעה",
]

HE_SERVICES = [
    "ביט", "פייבוקס", "בנק לאומי", "בנק הפועלים", "בנק דיסקונט", "בנק מזרחי טפחות",
    "מכבי", "כללית", "מאוחדת", "לאומית", "וולט", "תן ביס", "גט", "פנגו", "רב-קו", "רב־קו",
    "רכבת ישראל", "אגד", "דן", "ביטוח לאומי", "מס הכנסה", "רשות האוכלוסין", "דואר ישראל",
    "שופרסל", "רמי לוי", "ויקטורי", "אושר עד", "אמפם", "סופר-פארם", "איקאה", "יד2",
    "כביש 6", "חברת החשמל", "בזק", "פרטנר", "סלקום", "הוט", "פלאפון", "ישראכרט", "מקס", "כאל",
]

HE_ACRONYMS = [
    "צה״ל", "בע״מ", "מע״מ", "ד״ר", "ת״א", "אח״כ", "ב״ה", "כ״כ", "מ״ר", "ק״מ",
    "ח״ן", "ע״ש", "לפנה״צ", "אחה״צ", "דו״ח", "מו״מ", "מנכ״ל", "סמנכ״ל", "יו״ר", "ח״כ",
    "ע״י", "וכו׳", "מס׳", "טל׳", "רח׳", "ג׳", "ז׳", "צ׳", "ש״ח", "ארה״ב", "או״ם",
]

HE_UI_LABELS = [
    "שלח", "ביטול", "אישור", "הבא", "חזור", "חיפוש", "הגדרות", "פרופיל", "התראות",
    "הודעה חדשה", "כתוב הודעה...", "נראה לאחרונה היום", "מחובר", "מקליד...", "נקרא",
    "היום", "אתמול", "מחר", "עוד", "שתף", "מחק", "ערוך", "העתק", "הדבק", "שמור",
    "טוען...", "נסה שוב", "אין חיבור לאינטרנט", "התחבר", "התנתק", "הרשמה", "סיסמה",
    "אימות דו-שלבי", "קוד אימות", "תשלום", "יתרה", "העברה בנקאית", "חשבונית מס",
    "סיכום הזמנה", "משלוח חינם", "בתהליך", "הושלם", "בוטל", "ממתין לאישור",
    "יום ראשון", "יום שני", "יום שלישי", "יום רביעי", "יום חמישי", "יום שישי", "שבת",
]

HE_DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
HE_MONTHS = [
    "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
    "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
]

HE_WORDS_COMMON = [
    "שלום", "היי", "בוקר טוב", "ערב טוב", "לילה טוב", "תודה רבה", "בבקשה", "סליחה",
    "מעולה", "סבבה", "בסדר גמור", "אין בעיה", "מגניב", "אחלה", "יאללה", "נדבר",
    "מתי", "איפה", "כמה", "למה", "איך", "אפשר", "צריך", "רוצה", "יכול", "הולך",
    "עבודה", "פגישה", "משרד", "בית", "קפה", "ארוחת צהריים", "חופשה", "טיסה", "רכבת",
    "אוטובוס", "חניה", "פקקים", "משלוח", "חבילה", "תור", "רופא", "מרשם", "בדיקה",
    "חוזה", "שכירות", "ארנונה", "חשמל", "מים", "אינטרנט", "טלפון", "מסך", "סוללה",
]

TECH_TERMS = [
    "GitHub", "GitLab", "Slack", "Zoom", "Google Meet", "Google Drive", "Google Calendar",
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform", "Grafana", "Prometheus",
    "ClickHouse", "PostgreSQL", "Redis", "Kafka", "Airflow", "Nginx", "HAProxy", "Cloudflare",
    "Android", "Kotlin", "Python", "PyTorch", "ONNX", "CUDA", "Linux", "Ubuntu", "macOS",
    "Figma", "Jira", "Linear", "Notion", "Confluence", "WhatsApp", "Telegram", "Gmail",
    "ChatGPT", "Claude", "Gemini", "HuggingFace", "VS Code", "Android Studio", "Gradle",
    "PR", "API", "SDK", "CI", "CD", "UI", "UX", "SQL", "JSON", "YAML", "REST", "gRPC",
    "push", "pull", "merge", "rebase", "commit", "branch", "deploy", "build", "release",
    "hotfix", "review", "debug", "refactor", "pipeline", "timeout", "latency", "benchmark",
    "production", "staging", "dev", "prod-eu-2", "main", "feature/hebrew", "v2.4.1",
]

HE_PREFIXES = ["ל-", "ב-", "מ-", "ה-", "ו-", "ש-", "כ-"]

RU_FIRST_NAMES = [
    "Алексей", "Лёша", "Дмитрий", "Дима", "Иван", "Сергей", "Андрей", "Михаил", "Тимур",
    "Максим", "Артём", "Никита", "Егор", "Антон", "Владимир", "Роман", "Илья", "Кирилл",
    "Анна", "Мария", "Елена", "Ольга", "Наталья", "Татьяна", "Екатерина", "Юлия", "Дарья",
    "Алиса", "Полина", "Ксения", "София", "Виктория", "Вероника", "Марина", "Светлана",
]

RU_LAST_NAMES = [
    "Козлов", "Галиев", "Смирнов", "Иванов", "Кузнецов", "Попов", "Соколов", "Лебедев",
    "Новиков", "Морозов", "Петров", "Волков", "Соловьёв", "Васильев", "Зайцев", "Павлов",
    "Семёнов", "Голубев", "Виноградов", "Богданов", "Воробьёв", "Фёдоров", "Михайлов",
]

RU_UI_LABELS = [
    "Отправить", "Отмена", "Готово", "Далее", "Назад", "Поиск", "Настройки", "Профиль",
    "Сообщение", "Новое сообщение", "Печатает...", "В сети", "Был(а) недавно", "Сегодня",
    "Вчера", "Завтра", "Удалить", "Изменить", "Копировать", "Вставить", "Поделиться",
    "Сохранить", "Загрузка...", "Повторить", "Нет подключения", "Войти", "Выйти",
    "Уведомления", "Звонки", "Чаты", "Контакты", "Архив", "Избранное", "Корзина",
]

EN_CODE_SNIPPETS = [
    "git checkout -b feature/hebrew-ocr",
    "pytest -q ml/ocr_train/tests",
    "python3 -m ml.ocr.run screen.png --detail",
    "docker compose up -d --build",
    "kubectl get pods -n prod-eu-2",
    "SELECT count(*) FROM events WHERE lang = 'he';",
    "export CUDA_VISIBLE_DEVICES=0",
    "pip install torch onnxruntime python-bidi",
    "HTTP 200 OK (latency: 14.2 ms)",
    "ERROR: connection reset by peer (retry 3/5)",
    "WARNING: deprecated flag --legacy-mode",
    "val result = pipeline.run(image, control)",
    "def ctc_decode(probs: np.ndarray) -> tuple[str, float]:",
    "https://github.com/org/murmur/pull/4182",
    "ssh -p 22022 root@gpu-5090.internal",
    "sha256: 5371ee1ddaa7983cc62d0818d99e982b",
    "0.9842 accuracy | 0.0041 CER | 8.2 MB",
    "batch_size=256 lr=3e-4 amp=bf16 epochs=25",
    "README.md  config.py  pipeline.py  export_onnx.py",
    "support@example.co.il  dev-team@murmur.app",
]


def _rand_time(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, rng.randint(0, 59)]):02d}"


def _rand_date(rng: random.Random) -> str:
    d = rng.randint(1, 28)
    m = rng.randint(1, 12)
    y = rng.choice([2024, 2025, 2026])
    return rng.choice([f"{d:02d}/{m:02d}/{y}", f"{d:02d}.{m:02d}.{y}", f"{d:02d}/{m:02d}"])


def _rand_phone_il(rng: random.Random) -> str:
    prefix = rng.choice(["050", "052", "053", "054", "055", "058", "03", "02", "04", "08", "09"])
    if len(prefix) == 3:
        return f"{prefix}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    return f"{prefix}-{rng.randint(500, 999)}-{rng.randint(1000, 9999)}"


def _rand_shekel(rng: random.Random) -> str:
    val = rng.choice([
        f"{rng.randint(5, 99)}",
        f"{rng.randint(10, 499)}.{rng.choice(['00', '50', '90', '99'])}",
        f"{rng.randint(100, 950)}",
        f"{rng.randint(1, 25)},{rng.randint(100, 999)}",
    ])
    return rng.choice([f"{val} ₪", f"₪{val}", f"₪ {val}", f"{val} ש״ח"])


def _rand_pr(rng: random.Random) -> str:
    return f"#{rng.randint(10, 9999)}"


# ----------------------------------------------------------------------------- procedural generators


def gen_he_line(rng: random.Random) -> str:
    kind = rng.randint(0, 14)
    fn = rng.choice(HE_FIRST_NAMES)
    ln = rng.choice(HE_LAST_NAMES)
    city = rng.choice(HE_CITIES)
    street = rng.choice(HE_STREETS)
    svc = rng.choice(HE_SERVICES)
    acr = rng.choice(HE_ACRONYMS)
    t = _rand_time(rng)
    dt = _rand_date(rng)
    price = _rand_shekel(rng)
    day = rng.choice(HE_DAYS)
    mon = rng.choice(HE_MONTHS)

    if kind == 0:
        return rng.choice(HE_UI_LABELS)
    if kind == 1:
        return rng.choice([
            f"{fn} {ln}",
            f"{fn} {ln} · {t}",
            f"ד״ר {fn} {ln} ({city})",
            f"{fn}: {rng.choice(HE_WORDS_COMMON)}!",
        ])
    if kind == 2:
        return rng.choice([
            f"הועברו {price} באמצעות {svc}",
            f"תשלום בסך {price} ל{svc} אושר",
            f"מחיר: {price} (כולל מע״מ)",
            f"יתרה בחשבון {svc}: {price}",
            f"חשבונית מס קבלה מס׳ {rng.randint(10000, 99999)} — {price}",
            f"הזמנה מ{svc} בסך {price} תגיע ב-{t}",
        ])
    if kind == 3:
        return rng.choice([
            f"הפגישה ביום {day} בשעה {t} ב{city}",
            f"נקבע לך תור ל{svc} ליום {day}, {rng.randint(1, 28)} ב{mon} ב-{t}",
            f"תזכורת: שיחה עם {fn} בשעה {t}",
            f"רכבת מ{city} יוצאת ב-{t} מרציף {rng.randint(1, 8)}",
            f"עדכון משלוח: החבילה הגיעה לסניף {svc} ב{city}",
        ])
    if kind == 4:
        return rng.choice([
            f"{street} {rng.randint(1, 180)}, {city}",
            f"כתובת: {street} {rng.randint(1, 120)}, דירה {rng.randint(1, 40)}, {city}",
            f"טלפון ליצירת קשר: {_rand_phone_il(rng)}",
            f"קוד אימות חד-פעמי מ{svc}: {rng.randint(100000, 999999)}",
        ])
    if kind == 5:
        return rng.choice([
            f"היי {fn}, מה נשמע? אפשר לדבר ב-{t}?",
            f"שלום {fn}, שלחתי לך את המסמך ב{svc}",
            f"תודה רבה {fn}! נתראה ביום {day} ב{city}",
            f"סליחה על העיכוב, אני בדרך ל{city}, מגיע עוד {rng.randint(5, 45)} דקות",
            f"אפשר להעביר {price} ב{svc}?",
            f"בסדר גמור, נבדוק את זה מחר בבוקר ב-{t}",
        ])
    if kind == 6:
        return rng.choice([
            f"דו״ח חודשי ({mon} {rng.choice([2025, 2026])}) — {acr}",
            f"חברת {ln} בע״מ · ח״פ {rng.randint(510000000, 519999999)}",
            f"אישור תושב {city} · תאריך {dt}",
            f"מרחק נסיעה: {rng.randint(2, 180)} ק״מ · זמן משוער {rng.randint(10, 95)} דק׳",
            f"שטח הדירה: {rng.randint(35, 160)} מ״ר, {rng.randint(2, 5)} חדרים ב{city}",
        ])
    if kind == 7:
        # Exercise every Hebrew letter including sofits ך ם ן ף ץ and punctuation ׳ ״ ־ ₪
        return rng.choice([
            f"קיצוץ מחירים: סוף עונה רק ב-{price}!",
            f"פץ, מץ, נוץ, דף, עץ, קץ, ארנב, חתול, ג׳ירפה וצ׳יפס",
            f"כך גם היום: תן לי דקה, אבדוק את הדף ואשלח לך",
            f"צ׳ק על סך {price} לפקודת {fn} {ln}",
            f"בית־ספר, תל־אביב, רב־קו, על־ידי, כל־כך ({dt})",
            f"אבגדהוזחטיךכלםמןנסעףפץצקרשת",
            f"ז׳אנר, ג׳ינס, צ׳אט, פיג׳מה, פנצ׳ר, ג׳וקר ({acr})",
        ])
    # Short natural combinations of 2..6 Hebrew words
    words = [rng.choice(HE_WORDS_COMMON) for _ in range(rng.randint(2, 5))]
    if rng.random() < 0.3:
        words.append(rng.choice([city, svc, price, t, acr]))
    return " ".join(words)


def gen_he_en_line(rng: random.Random) -> str:
    fn = rng.choice(HE_FIRST_NAMES)
    tech = rng.choice(TECH_TERMS)
    tech2 = rng.choice(TECH_TERMS)
    pref = rng.choice(HE_PREFIXES)
    t = _rand_time(rng)
    pr = _rand_pr(rng)
    price = _rand_shekel(rng)

    templates = [
        f"עשיתי push {pref}{tech}",
        f"פתחתי PR {pr} {pref}{tech}, אשמח ל-review",
        f"שלחתי לך הודעה {pref}{tech}",
        f"ה-build {pref}{tech} נפל על timeout",
        f"נפגש {pref}{tech} בשעה {t}",
        f"תבדוק את ה-logs {pref}{tech} בבקשה",
        f"עדכנתי את ה-config {pref}{tech2} ועשיתי deploy",
        f"היי {fn}, אפשר גישה {pref}{tech}?",
        f"צריך לעשות rebase על main לפני ה-merge",
        f"הוספתי תמיכה {pref}Hebrew ב-{tech}",
        f"תיקנתי את ה-bug ב-{tech} ({pr})",
        f"ה-latency ירד ל-{rng.randint(4, 120)}ms אחרי האופטימיזציה",
        f"שיתפתי את המסמך {pref}{tech} עם {fn}",
        f"חיוב חודשי {tech}: {price} ({_rand_date(rng)})",
        f"עובד על feature/{rng.choice(['ocr', 'hebrew', 'asr', 'sync'])} ב-{tech}",
        f"אפשר לעלות ל-call קצר {pref}{tech} ב-{t}?",
        f"{tech} {pref}{tech2} — סטטוס: OK ({t})",
        f"התקנתי {tech} v{rng.randint(1, 9)}.{rng.randint(0, 15)} על {rng.choice(['Android', 'Linux', 'macOS'])}",
        f"בדקתי ב-{tech}, הכל ירוק ב-CI",
        f"תשלח לי את הלינק {pref}{tech} ב-WhatsApp",
    ]
    return rng.choice(templates)


def gen_ru_line(rng: random.Random, with_en: bool = False) -> str:
    fn = rng.choice(RU_FIRST_NAMES)
    ln = rng.choice(RU_LAST_NAMES)
    tech = rng.choice(TECH_TERMS)
    tech2 = rng.choice(TECH_TERMS)
    t = _rand_time(rng)
    dt = _rand_date(rng)
    pr = _rand_pr(rng)

    if with_en:
        templates = [
            f"Привет! Скинь ссылку на дашборд в {tech}",
            f"Ещё нужен доступ к {tech} кластеру {rng.choice(['prod-eu-2', 'staging-1', 'us-east'])}?",
            f"Ревью PR {pr} назначил на {fn} {ln}",
            f"Собрал новый build для {tech}, можно тестить",
            f"В {tech} снова упал pipeline после merge {pr}",
            f"Созвон в {tech} перенесён на {t}",
            f"Сделал push в ветку feature/{rng.choice(['hebrew', 'ocr', 'ui', 'fix'])}",
            f"Проверь логи в {tech} и метрики в {tech2}",
            f"Обновил зависимости {tech} до версии {rng.randint(1, 5)}.{rng.randint(0, 20)}.0",
            f"Экспортировал модель в ONNX ({rng.randint(4, 18)}.{rng.randint(0, 9)} MB) без ошибок",
            f"Отправил инвайт в {tech}, время {t}",
            f"{fn}: посмотри, пожалуйста, issue {pr} в {tech}",
            f"Запустил бенчмарк на {tech} — latency {rng.randint(5, 80)} мс",
            f"Конфиг {tech} лежит в папке ml/ocr/config.py",
        ]
        return rng.choice(templates)

    templates = [
        rng.choice(RU_UI_LABELS),
        f"{fn} {ln}",
        f"Созвон в {t} перенесён на четверг",
        f"Привет, {fn}! Как дела? Во сколько сегодня встречаемся?",
        f"Отправил документы на почту, проверь пожалуйста",
        f"Оплата заказа №{rng.randint(1000, 99999)} на сумму {rng.randint(150, 15000)} ₽ прошла успешно",
        f"Напоминание: запись к врачу на {dt} в {t}",
        f"Буду через {rng.randint(5, 40)} минут, попал в пробку",
        f"Хорошо, договорились! Напишу завтра утром",
        f"Скинь, пожалуйста, номер телефона и адрес",
        f"Спасибо большое! Всё отлично работает",
        f"Посылка прибыла в пункт выдачи ({dt})",
        f"Код подтверждения: {rng.randint(1000, 999999)} (никому не сообщайте)",
        f"«Обновление завершено» — перезапустите приложение",
        f"Встреча команды: {dt}, {t} — переговорка №{rng.randint(1, 9)}",
    ]
    return rng.choice(templates)


def gen_en_line(rng: random.Random) -> str:
    kind = rng.randint(0, 4)
    if kind == 0:
        return rng.choice(EN_CODE_SNIPPETS)
    tech = rng.choice(TECH_TERMS)
    tech2 = rng.choice(TECH_TERMS)
    t = _rand_time(rng)
    pr = _rand_pr(rng)
    templates = [
        f"Merged pull request {pr} into main ({t})",
        f"Deploying {tech} service to {rng.choice(['prod-eu-2', 'us-west-2', 'il-central-1'])}",
        f"Build #{rng.randint(100, 9999)} passed in {rng.randint(12, 240)}s",
        f"Meeting with Engineering team at {t} via {tech}",
        f"Latency p99: {rng.randint(5, 190)} ms | Error rate: 0.{rng.randint(0, 9):02d}%",
        f"Updated {tech} integration with {tech2} connector",
        f"Invoice #{rng.randint(10000, 99999)} — ${rng.randint(10, 999)}.{rng.choice(['00', '99'])} paid",
        f"Download complete: model_v{rng.randint(1, 6)}.onnx ({rng.randint(2, 99)}.{rng.randint(0, 9)} MB)",
        f"Search messages, files, and channels...",
        f"Last seen today at {t}",
    ]
    return rng.choice(templates)


def gen_ru_he_line(rng: random.Random) -> str:
    ru_fn = rng.choice(RU_FIRST_NAMES)
    he_fn = rng.choice(HE_FIRST_NAMES)
    city = rng.choice(HE_CITIES)
    svc = rng.choice(HE_SERVICES)
    price = _rand_shekel(rng)
    t = _rand_time(rng)
    dt = _rand_date(rng)

    templates = [
        "Привет שלום",
        f"Привет {he_fn}, שלום! Как дела?",
        f"שלום {ru_fn}, увидимся в {t}",
        f"Встретимся в {city} сегодня в {t}",
        f"Оплатил через {svc} {price}",
        f"Перевёл {price} на {svc}, проверь",
        f"Записался в {svc} на {dt} в {t}",
        f"Еду на {svc} в {city}, буду к {t}",
        f"Снял квартиру в {city} за {rng.randint(4, 12)},{rng.choice(['000', '500'])} ₪",
        f"Отправил посылку через {svc} ({city})",
        f"Нужен טופס 101 и תלוש משכורת для бухгалтерии",
        f"Посмотри объявление на יד2 в районе {city}",
        f"Купил продукты в {svc} на {price}",
        f"שלחתי לך сообщение в Телеграм",
        f"תודה большое, {ru_fn}! До завтра",
        f"הפגישה в {t} ב{city}",
        f"Скинь свой номер ת״ז и адрес в {city}",
        f"Получил письмо из ביטוח לאומי ({dt})",
        f"Заказал еду через וולט — доставка {price}",
        f"Работаю в офисе в {city}, метро / רכבת рядом",
    ]
    return rng.choice(templates)


# ----------------------------------------------------------------------------- HF corpus streaming


_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[.!?。])\s+")


def _chunk_text_to_lines(
    text: str,
    cs: Charset,
    rng: random.Random,
    min_len: int = 4,
    max_len: int = 64,
) -> Iterator[str]:
    """Slice long prose paragraphs into realistic single-line UI crops (4..64 chars)."""
    text = strip_nikkud(text)
    for sent in _SPLIT_RE.split(text):
        cleaned = cs.filter_text(sent)
        if len(cleaned) < min_len:
            continue
        if len(cleaned) <= max_len:
            yield cleaned
            continue
        words = cleaned.split(" ")
        cur: list[str] = []
        cur_len = 0
        target = rng.randint(24, max_len)
        for w in words:
            if not w:
                continue
            add = len(w) + (1 if cur else 0)
            if cur and cur_len + add > target:
                line = " ".join(cur)
                if min_len <= len(line) <= max_len:
                    yield line
                cur = [w]
                cur_len = len(w)
                target = rng.randint(24, max_len)
            else:
                cur.append(w)
                cur_len += add
        if cur:
            line = " ".join(cur)
            if min_len <= len(line) <= max_len:
                yield line


def stream_hf_wikipedia(lang: str, limit: int, cs: Charset, rng: random.Random) -> list[str]:
    """Stream lines from HuggingFace `wikimedia/wikipedia` (`20231101.<lang>`). Returns [] if unavailable."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(f"[build_corpus] `datasets` not installed; skipping HF wikipedia ({lang})", flush=True)
        return []

    out: list[str] = []
    configs = [f"20231101.{lang}"]
    for cfg in configs:
        try:
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            for row in ds:
                txt = row.get("text") or ""
                for line in _chunk_text_to_lines(txt, cs, rng):
                    if lang == "he" and not has_hebrew(line):
                        continue
                    out.append(line)
                    if len(out) >= limit:
                        print(f"[build_corpus] HF wikimedia/wikipedia {cfg}: collected {len(out):,} lines", flush=True)
                        return out
            break
        except Exception as e:
            print(f"[build_corpus] HF wikimedia/wikipedia {cfg} unavailable ({e}); continuing", flush=True)
    return out


def stream_hf_ivrit(limit: int, cs: Charset, rng: random.Random) -> list[str]:
    """Stream conversational Hebrew transcripts from public ivrit-ai datasets if accessible."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return []

    candidates = [
        ("ivrit-ai/crowd-transcribe-v5", None, "train", ("text", "transcript", "sentence")),
        ("ivrit-ai/whisper-training", None, "train", ("text", "transcript", "sentence")),
    ]
    out: list[str] = []
    for name, cfg, split, cols in candidates:
        try:
            ds = load_dataset(name, cfg, split=split, streaming=True)
            for row in ds:
                txt = ""
                for c in cols:
                    if isinstance(row.get(c), str) and row[c].strip():
                        txt = row[c]
                        break
                if not txt:
                    continue
                for line in _chunk_text_to_lines(txt, cs, rng):
                    if has_hebrew(line):
                        out.append(line)
                        if len(out) >= limit:
                            print(f"[build_corpus] HF {name}: collected {len(out):,} lines", flush=True)
                            return out
            if out:
                print(f"[build_corpus] HF {name}: collected {len(out):,} lines", flush=True)
                return out
        except Exception as e:
            print(f"[build_corpus] HF {name} skipped ({e})", flush=True)
    return out


def load_local_text_lines(paths: Iterable[Path], cs: Charset, rng: random.Random, limit: int) -> list[str]:
    out: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                for line in _chunk_text_to_lines(raw, cs, rng):
                    out.append(line)
                    if len(out) >= limit:
                        return out
    return out


# ----------------------------------------------------------------------------- corpus assembly


def _make_record(rec_id: int, text: str, cat: str, rng: random.Random) -> dict:
    auto_dir = detect_base_dir(text)
    # On mixed lines (he_en / ru_he), 85% natural first-strong direction, 15% opposite app-UI base_dir
    if cat in ("he_en", "ru_he") and has_hebrew(text) and rng.random() < 0.15:
        base_dir = "L" if auto_dir == "R" else "R"
    else:
        base_dir = auto_dir
    vis = to_visual_order(text, base_dir=base_dir)
    return {
        "id": rec_id,
        "text": text,
        "visual": vis,
        "base_dir": base_dir,
        "cat": cat,
    }


def _fill_category(
    target_n: int,
    seed_pool: list[str],
    gen_fn,
    cs: Charset,
    seen: set[str],
    rng: random.Random,
    max_len: int = 72,
) -> list[str]:
    """Fill `target_n` unique valid lines from `seed_pool` first, then `gen_fn` (allowing mild variation if exhausted)."""
    rng.shuffle(seed_pool)
    picked: list[str] = []
    for raw in seed_pool:
        line = cs.filter_text(raw)
        if not (2 <= len(line) <= max_len):
            continue
        if line in seen:
            continue
        seen.add(line)
        picked.append(line)
        if len(picked) >= target_n:
            return picked

    stale = 0
    while len(picked) < target_n:
        line = cs.filter_text(gen_fn(rng))
        if not (2 <= len(line) <= max_len):
            stale += 1
            if stale > 50_000:
                break
            continue
        if line in seen:
            stale += 1
            # If procedural space saturates in huge offline runs, allow controlled repeats with distinct styling later
            if stale > 10_000:
                picked.append(line)
                continue
            continue
        stale = 0
        seen.add(line)
        picked.append(line)
    return picked


def build_corpus(
    out_dir: Path,
    num_lines: int = 400_000,
    val_ratio: float = 0.02,
    max_val: int = 8_000,
    seed: int = 42,
    offline: bool = False,
    dict_path: Path = DEFAULT_DICT_PATH,
    extra_he: list[Path] | None = None,
    extra_ru: list[Path] | None = None,
    extra_en: list[Path] | None = None,
) -> dict:
    """Build balanced train/val JSONL manifests and return summary stats."""
    cs = load_charset(dict_path)
    rng = random.Random(seed)

    # Target proportions: 40% he, 15% he_en, 22% ru, 13% ru_en (35% total ru+ru_en), 5% en, 5% ru_he
    targets = {
        "he": int(round(num_lines * 0.40)),
        "he_en": int(round(num_lines * 0.15)),
        "ru": int(round(num_lines * 0.22)),
        "ru_en": int(round(num_lines * 0.13)),
        "en": int(round(num_lines * 0.05)),
        "ru_he": 0,
    }
    targets["ru_he"] = max(1, num_lines - sum(targets.values()))

    he_pool: list[str] = []
    ru_pool: list[str] = []
    en_pool: list[str] = []

    if extra_he:
        he_pool.extend(load_local_text_lines(extra_he, cs, rng, targets["he"]))
    if extra_ru:
        ru_pool.extend(load_local_text_lines(extra_ru, cs, rng, targets["ru"]))
    if extra_en:
        en_pool.extend(load_local_text_lines(extra_en, cs, rng, targets["en"]))

    if not offline:
        # Reserve ~25% of `he` for Israeli names/prices/services/UI templates; pull ~75% from HF
        hf_he_want = int(targets["he"] * 0.75) - len(he_pool)
        if hf_he_want > 0:
            ivrit = stream_hf_ivrit(int(hf_he_want * 0.35), cs, rng)
            he_pool.extend(ivrit)
            wiki_he = stream_hf_wikipedia("he", max(0, hf_he_want - len(ivrit)), cs, rng)
            he_pool.extend(wiki_he)
        hf_ru_want = int(targets["ru"] * 0.75) - len(ru_pool)
        if hf_ru_want > 0:
            ru_pool.extend(stream_hf_wikipedia("ru", hf_ru_want, cs, rng))
        hf_en_want = int(targets["en"] * 0.65) - len(en_pool)
        if hf_en_want > 0:
            en_pool.extend(stream_hf_wikipedia("en", hf_en_want, cs, rng))

    # Also synthesize realistic he_en and ru_en by injecting English tech terms into HF prose slices
    he_en_pool: list[str] = []
    for s in he_pool[: min(len(he_pool), targets["he_en"] // 2)]:
        words = s.split(" ")
        if len(words) >= 3:
            idx = rng.randint(1, len(words) - 1)
            tok = rng.choice(HE_PREFIXES) + rng.choice(TECH_TERMS) if rng.random() < 0.6 else rng.choice(TECH_TERMS)
            cand = " ".join(words[:idx] + [tok] + words[idx:])
            if len(cand) <= 68:
                he_en_pool.append(cand)

    ru_en_pool: list[str] = []
    for s in ru_pool[: min(len(ru_pool), targets["ru_en"] // 2)]:
        words = s.split(" ")
        if len(words) >= 3:
            idx = rng.randint(1, len(words) - 1)
            cand = " ".join(words[:idx] + [rng.choice(TECH_TERMS)] + words[idx:])
            if len(cand) <= 68:
                ru_en_pool.append(cand)

    seen: set[str] = set()
    by_cat: dict[str, list[str]] = {
        "he": _fill_category(targets["he"], he_pool, gen_he_line, cs, seen, rng),
        "he_en": _fill_category(targets["he_en"], he_en_pool, gen_he_en_line, cs, seen, rng),
        "ru": _fill_category(targets["ru"], ru_pool, lambda r: gen_ru_line(r, with_en=False), cs, seen, rng),
        "ru_en": _fill_category(targets["ru_en"], ru_en_pool, lambda r: gen_ru_line(r, with_en=True), cs, seen, rng),
        "en": _fill_category(targets["en"], en_pool, gen_en_line, cs, seen, rng),
        "ru_he": _fill_category(targets["ru_he"], [], gen_ru_he_line, cs, seen, rng),
    }

    # Stratified split ensuring zero text overlap between train and val
    total_val = min(max_val, max(len(by_cat) * 10, int(round(num_lines * val_ratio))))
    train_pairs: list[tuple[str, str]] = []
    val_pairs: list[tuple[str, str]] = []

    for cat, lines in by_cat.items():
        rng.shuffle(lines)
        cat_val = max(1, int(round(total_val * (len(lines) / max(1, num_lines)))))
        cat_val = min(cat_val, max(1, len(lines) // 5))
        # Deduplicate strictly across split boundary by text
        val_set = set(lines[:cat_val])
        val_lines = list(val_set)
        train_lines = [ln for ln in lines[cat_val:] if ln not in val_set]
        val_pairs.extend((ln, cat) for ln in val_lines)
        train_pairs.extend((ln, cat) for ln in train_lines)

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_corpus.jsonl"
    val_path = out_dir / "val_corpus.jsonl"

    train_counts = Counter()
    with train_path.open("w", encoding="utf-8") as f:
        for i, (txt, cat) in enumerate(train_pairs):
            rec = _make_record(i, txt, cat, rng)
            if not cs.can_encode(rec["visual"]):
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            train_counts[cat] += 1

    val_counts = Counter()
    with val_path.open("w", encoding="utf-8") as f:
        for i, (txt, cat) in enumerate(val_pairs):
            rec = _make_record(i, txt, cat, rng)
            if not cs.can_encode(rec["visual"]):
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            val_counts[cat] += 1

    stats = {
        "train_total": sum(train_counts.values()),
        "val_total": sum(val_counts.values()),
        "train_by_cat": dict(train_counts),
        "val_by_cat": dict(val_counts),
        "offline": offline,
        "seed": seed,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (out_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build balanced 3-language (ru+en+he) OCR text corpus.")
    ap.add_argument("--out-dir", type=Path, default=Path(".cache/ml/ocr_train/corpus"))
    ap.add_argument("--num-lines", type=int, default=400_000, help="Total lines (default 400,000)")
    ap.add_argument("--val-ratio", type=float, default=0.02, help="Validation fraction (default 0.02)")
    ap.add_argument("--max-val", type=int, default=8_000, help="Cap validation size for fast eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true", help="Do not stream from HuggingFace")
    ap.add_argument("--smoke", action="store_true", help="Fast 2,000-line offline smoke corpus")
    ap.add_argument("--dict", type=Path, default=DEFAULT_DICT_PATH)
    ap.add_argument("--extra-he", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-ru", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-en", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    num_lines = 2_000 if args.smoke else args.num_lines
    val_ratio = 0.10 if args.smoke else args.val_ratio
    offline = True if args.smoke else args.offline

    stats = build_corpus(
        out_dir=args.out_dir,
        num_lines=num_lines,
        val_ratio=val_ratio,
        max_val=args.max_val,
        seed=args.seed,
        offline=offline,
        dict_path=args.dict,
        extra_he=args.extra_he,
        extra_ru=args.extra_ru,
        extra_en=args.extra_en,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
