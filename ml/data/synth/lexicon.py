"""Entity lexicon for the synthetic pipeline.

Every entity has a split (train/dev/test) derived from a hash of its text, so
held-out eval scenarios use names, terms and distractor pairs never seen in
training (leakage-free by construction). `spoken_ru` lists how a Russian
speaker says a Latin-script term (used by the ASR channel to produce
"клик хаус" for ClickHouse).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ml.common.util import bucket_split

ENTITY_DEV, ENTITY_TEST = 0.12, 0.12  # share of entities reserved per held-out split


@dataclass(frozen=True)
class Entity:
    text: str                      # canonical written form (as on screen / in target)
    kind: str                      # person_ru | person_en | tech | company | place_ru | place_en | product
    lang: str                      # script of the canonical form: ru | en
    spoken_ru: tuple = ()          # Cyrillic spoken forms for Latin terms in Russian speech
    gender: str = ""               # m | f for people
    known: bool = False            # well-known term: a model may restore it without context

    @property
    def split(self) -> str:
        return bucket_split("entity:" + self.text, dev=ENTITY_DEV, test=ENTITY_TEST)


RU_MALE = "Алексей Андрей Артём Борис Вадим Виктор Владимир Глеб Григорий Даниил Денис Дмитрий Евгений Егор Иван Игорь Илья Кирилл Константин Лев Максим Марк Матвей Михаил Никита Николай Олег Павел Пётр Роман Руслан Семён Сергей Станислав Степан Тимофей Тимур Фёдор Юрий Ярослав Арсений Эльдар Ринат Айдар".split()
RU_FEMALE = "Алёна Алина Алиса Анастасия Анна Валерия Вера Виктория Галина Дарья Диана Ева Екатерина Елена Елизавета Жанна Зарина Ирина Камила Карина Ксения Лариса Лилия Любовь Марина Мария Милана Надежда Наталья Нина Ольга Полина Светлана София Таисия Татьяна Ульяна Юлия Яна Эльвира Регина Асель".split()
RU_SURNAMES_OV = "Козлов Галиев Смирнов Волков Соколов Лебедев Новиков Морозов Петров Павлов Семёнов Голубев Виноградов Богданов Воробьёв Фёдоров Михайлов Беляев Тарасов Белов Комаров Орлов Киселёв Макаров Андреев Ковалёв Ильин Гусев Титов Кузьмин Кудрявцев Баранов Куликов Алексеев Степанов Яковлев Сорокин Сергеев Романов Захаров Борисов Королёв Герасимов Пономарёв Григорьев Лазарев Медведев Ершов Никитин Соболев Рябов Поляков Цветков Данилов Жуков Фролов Журавлёв Николаев Крылов Максимов Сидоров Осипов Белоусов Федотов Дорофеев Егоров Матвеев Бобров Дмитриев Калинин Анисимов Петухов Антонов Тимофеев Никифоров Веселов Филиппов Марков Большаков Суханов Миронов Ширяев Александров Коновалов Шестаков Казаков Ефимов Денисов Громов Фомин Давыдов Мельников Щербаков Блинов Колесников Карпов Афанасьев Власов Маслов Исаков Тихонов Аксёнов Гаврилов Родионов Котов Горбунов Кудряшов Быков Зуев Третьяков Савельев Панов Рыбаков Суворов Абрамов Воронов Мухин Архипов Трофимов Мартынов Емельянов Горшков Чернов Овчинников Селезнёв Панфилов Копылов Михеев Галкин Назаров Лобанов Лукин Беляков Потапов Некрасов Хохлов Жданов Наумов Шилов Воронцов Ермаков Дроздов Игнатьев Савин Логинов Сафонов Капустин Кириллов Моисеев Елисеев Кошелев Костин Горбачёв Орехов Ефремов Исаев Евдокимов Калашников Кабанов Носков Юдин Кулагин Лапин Прохоров Нестеров Харитонов Агафонов Муравьёв Ларионов Федосеев Зимин Пахомов Шубин Игнатов Филатов Крюков Рогов Кулаков Терентьев Молчанов Владимиров Артемьев Гурьев Зиновьев Гришин Кононов Дементьев Ситников Симонов Мишин Фадеев Комиссаров Мамонтов Носов Гуляев Шаров Устинов Вишняков Евсеев Лаврентьев Брагин Константинов Корнилов Авдеев Зыков Бирюков Шарапов Никонов Щукин Дьячков Одинцов Сазонов Якушев Красильников Гордеев Самойлов Князев Беспалов Уваров Шашков Бобылёв Доронин Белозёров Рожков Самсонов Мясников Лихачёв Буров Сысоев Фомичёв Русаков Стрелков Гущин Тетерин Колобов Субботин Фокин Блохин Селиверстов Пестов Кондратьев Силин Меркушев Лыткин Туров Хабибуллин Шарипов Сафин Зарипов Валиев Гарипов Мухаметов Ахметов Исмаилов".split()
RU_SURNAMES_OTHER = "Шевченко Ковальчук Бондаренко Ткаченко Кравченко Лысенко Мельник Коваль Гончаренко Савченко Руденко Пак Ким Цой Остапчук Левченко Яременко Литвиненко Голуб Сокол Рудь Гринберг Шапиро Кац Левин Фридман Шульц Миллер Эйдельман Рахимов Юсупов Алиев Мамедов Гусейнов Каримов Назарбаев".split()
EN_FIRST = "Priya Jürgen Ingrid Siobhan Aoife Niamh Mateo Santiago Chloé Zoë Björn Søren Ananya Arjun Rohan Ravi Mei Wei Hiroshi Yuki Kenji Olusegun Chidi Ngozi Kwame Fatima Aisha Omar Yusuf Leila Dmitri Katarzyna Wojciech Małgorzata Thibault Gwendolyn Sean Hannah Lucas Emma Olivia Noah Liam Sophia Ethan Isabella Mason Ava Logan Harper Elijah Amelia James Charlotte Benjamin Mia Henry Evelyn Alexander Abigail Sebastian Emily Jack Ella Owen Scarlett Theodore Grace Nathaniel Madeleine Declan Saoirse Eoin Ciarán".split()
EN_SURNAMES = "Raghunathan Weiß Sørensen Nakamura Okonkwo Adeyemi Gallagher O'Sullivan McAllister Fitzgerald Schmidt Müller Schneider Fischer Wagner Becker Hoffmann Kowalski Nowak Wiśniewski Dubois Lefèvre Moreau Laurent Fontaine Rossi Bianchi Esposito García Fernández Rodríguez Martínez Hernández López González Pérez Sánchez Ramírez Torres Nguyen Tran Pham Chen Wang Zhang Liu Kim Park Choi Patel Sharma Gupta Iyer Krishnamurthy Venkataraman Subramanian Chatterjee Mukherjee Banerjee Thompson Anderson Taylor Thomas Jackson White Harris Martin Clarke Robinson Walker Wright Mitchell Carter Phillips Evans Turner Torvalds Halvorsen Lindqvist Johansson Andersson Virtanen Korhonen Papadopoulos Georgiou Yilmaz Kaya Demir Abernathy Ellsworth Whitfield Pemberton Ashcroft Blackwood Hargreaves".split()

# Latin-script tech/product terms and how they are said in Russian speech.
TECH = {
    "ClickHouse": ["клик хаус", "кликхаус"], "Kubernetes": ["кубернетис", "кубернетес", "кубер"], "GitHub": ["гитхаб"],
    "GitLab": ["гитлаб"], "Slack": ["слак"], "Jira": ["джира", "жира"], "Confluence": ["конфлюенс"], "Grafana": ["графана"],
    "Prometheus": ["прометеус", "прометей"], "PostgreSQL": ["постгрес", "постгрескьюэль"], "Postgres": ["постгрес"],
    "Redis": ["редис"], "Kafka": ["кафка"], "Docker": ["докер"], "Helm": ["хелм"], "Terraform": ["терраформ"],
    "Ansible": ["ансибл"], "Airflow": ["эйрфлоу", "аирфлоу"], "Figma": ["фигма"], "Notion": ["ноушн", "ношн"],
    "Telegram": ["телеграм", "телега"], "WhatsApp": ["вотсап", "ватсап"], "Zoom": ["зум"], "Google Meet": ["гугл мит"],
    "Python": ["питон", "пайтон"], "TypeScript": ["тайпскрипт"], "JavaScript": ["джаваскрипт"], "React": ["реакт"],
    "Kotlin": ["котлин"], "Android": ["андроид"], "iOS": ["ай ос", "иос"], "Linux": ["линукс"], "macOS": ["мак ос"],
    "Windows": ["виндоус", "винда"], "AWS": ["эй дабл ю эс", "авс"], "S3": ["эс три"], "Yandex Cloud": ["яндекс клауд"],
    "Elasticsearch": ["эластиксёрч", "эластик"], "MongoDB": ["монго", "монгодиби"], "MySQL": ["майэскьюэль", "мускул"],
    "Nginx": ["энджинкс", "нджинкс"], "Jenkins": ["дженкинс"], "Sentry": ["сентри"], "Datadog": ["датадог"],
    "Kibana": ["кибана"], "Loki": ["локи"], "Tempo": ["темпо"], "Argo CD": ["арго сиди", "арго"], "Istio": ["истио"],
    "Envoy": ["энвой"], "gRPC": ["джи арписи", "грпц"], "GraphQL": ["графкуэль", "граф кью эль"], "REST API": ["рест апи"],
    "OAuth": ["оаус", "о ауф"], "JWT": ["джей дабл ю ти", "жвт"], "CI/CD": ["сиай сиди", "цикд"], "PyTorch": ["пайторч"],
    "TensorFlow": ["тензорфлоу"], "Hugging Face": ["хаггинг фейс"], "ChatGPT": ["чат джипити", "чатгпт"],
    "Claude": ["клод"], "Copilot": ["копайлот"], "VS Code": ["вэ эс код", "вскод"], "IntelliJ": ["интеллиджей"],
    "Xcode": ["икс код"], "Swift": ["свифт"], "Rust": ["раст"], "Golang": ["голанг", "гоу"], "Java": ["джава"],
    "Spring Boot": ["спринг бут"], "Django": ["джанго"], "FastAPI": ["фаст апи"], "Celery": ["селери"],
    "RabbitMQ": ["рэббит", "раббит эм кью"], "Vault": ["волт", "ваулт"], "Consul": ["консул"], "Karpenter": ["карпентер"],
    "Bitbucket": ["битбакет"], "Trello": ["трелло"], "Miro": ["миро"], "Excel": ["эксель"], "PowerPoint": ["пауэрпоинт"],
    "Google Docs": ["гугл докс"], "Google Sheets": ["гугл шитс", "гугл таблицы"], "Dropbox": ["дропбокс"],
    "OneDrive": ["ван драйв"], "Salesforce": ["сейлсфорс"], "HubSpot": ["хабспот"], "Tableau": ["табло"],
    "Power BI": ["пауэр би ай"], "Metabase": ["метабейс"], "Superset": ["суперсет"], "dbt": ["дибити"],
    "Snowflake": ["сноуфлейк"], "BigQuery": ["бигквери"], "Spark": ["спарк"], "Hadoop": ["хадуп"], "Flink": ["флинк"],
    "Ozon": ["озон"], "Wildberries": ["вайлдберриз", "вб"], "Avito": ["авито"], "Kaspi": ["каспи"],
    "Revolut": ["револют"], "PayPal": ["пэйпал"], "Stripe": ["страйп"], "Uber": ["убер"], "Airbnb": ["эйрбиэнби"],
    "Booking": ["букинг"], "Netflix": ["нетфликс"], "Spotify": ["спотифай"], "YouTube": ["ютуб"], "TikTok": ["тикток"],
    "Instagram": ["инстаграм", "инста"], "LinkedIn": ["линкедин"], "Reddit": ["реддит"], "Discord": ["дискорд"],
    "Steam": ["стим"], "PlayStation": ["плейстейшн"], "Xbox": ["иксбокс"], "Tesla": ["тесла"], "OpenAI": ["опен эй ай"],
    "NVIDIA": ["нвидиа", "энвидиа"], "Qualcomm": ["квалком"], "Samsung": ["самсунг"], "Galaxy Fold": ["гэлакси фолд"],
    "Pixel": ["пиксель"], "MacBook": ["макбук"], "iPhone": ["айфон"], "AirPods": ["эйрподс"], "Apple Watch": ["эппл вотч"],
    "Kubeflow": ["кубфлоу"], "MLflow": ["эмэльфлоу"], "Weights & Biases": ["вейтс энд байасес", "ванбд"],
    "LoRA": ["лора"], "vLLM": ["вилэлэм", "ви эл эл эм"], "llama.cpp": ["лама сипипи"], "GGUF": ["джи джи ю эф", "гуф"],
    "ONNX": ["оникс"], "staging": ["стейджинг", "стейдж"], "production": ["продакшн", "прод"], "deadline": ["дедлайн"],
    "code review": ["код ревью"], "pull request": ["пул реквест", "пулл реквест"], "merge request": ["мерж реквест"],
    "feature flag": ["фича флаг"], "hotfix": ["хотфикс"], "backlog": ["бэклог"], "standup": ["стендап"],
    "sprint": ["спринт"], "release notes": ["релиз ноутс"], "dashboard": ["дашборд"], "onboarding": ["онбординг"],
    "Helm chart": ["хелм чарт"],
}
KNOWN_TECH = {"GitHub", "Slack", "Telegram", "WhatsApp", "Zoom", "Python", "Android", "Linux", "Windows", "YouTube",
              "Instagram", "iPhone", "Excel", "PowerPoint", "ChatGPT", "Docker", "Kubernetes", "PostgreSQL", "Postgres",
              "Jira", "Figma", "Google Docs", "TikTok", "Netflix", "Spotify", "Samsung", "Ozon", "Avito", "Wildberries"}
# Russian-derived English slang stays Cyrillic in targets (rule 7 of the default prompt).
RU_SLANG = ["задеплоить", "задеплоил", "запушить", "запушил", "закоммитить", "закоммитил", "смержить", "смержил",
            "зарелизить", "зарелизили", "пофиксить", "пофиксил", "дебажить", "заапрувить", "заапрувил", "отревьюить",
            "засетапить", "законтрибьютить", "залогиниться", "расшарить", "расшарил", "скринить", "чекнуть", "апнуть"]
CODE_IDS = ["prod-eu-2", "kube-prometheus", "INV-0093", "MUR-1427", "api-gateway", "user_id", "feature/login-v2",
            "v2.3.1", "release-17", "PR #482", "OPS-311", "staging-ru-1", "payments-svc", "auth-service", "billing_v3",
            "DATA-2048", "hotfix/oom", "node-pool-b", "eu-central-1", "req_7f3a"]
COMPANIES = ["Сбер", "Т-Банк", "Яндекс", "VK", "Ozon", "Авито", "Касперский", "JetBrains", "Miro", "Acme Corp",
             "Northwind", "Globex", "Initech", "Contoso", "Fabrikam", "Umbrella Logistics", "Stark Industries",
             "ООО «Ромашка»", "АО «Вектор»", "ИП Соколова"]
PLACES_RU = ["Казань", "Новосибирск", "Екатеринбург", "Нижний Новгород", "Самара", "Пермь", "Тбилиси", "Ереван",
             "Алматы", "Белград", "Лимассол", "Сочи", "Калининград", "Владивосток", "Мурманск", "Тверь", "Ярославль",
             "Сокольники", "Патриаршие пруды", "Лефортово"]
PLACES_EN = ["Zurich", "Lisbon", "Reykjavik", "Edinburgh", "Ljubljana", "Wrocław", "Kraków", "Gdańsk", "Eindhoven",
             "Leuven", "Aarhus", "Tromsø", "Sacramento", "Albuquerque", "Worcester", "Leicester", "Gloucester",
             "Thessaloniki", "Valparaíso", "Montréal"]
EMAIL_DOMAINS = ["gmail.com", "yandex.ru", "mail.ru", "example.com", "outlook.com", "proton.me", "acme.io",
                 "company.ru", "uni.edu", "murmur.app"]
# confusable pairs for distractors (both real, both plausible on screen)
CONFUSABLE = [("ClickHouse", "ClickUp"), ("Grafana", "Graphite"), ("Kafka", "Karpenter"), ("Jira", "Jenkins"),
              ("Slack", "Stack Overflow"), ("Notion", "Motion"), ("Figma", "Framer"), ("Kibana", "Kubeflow"),
              ("Redis", "Rediska"), ("Helm", "Heroku"), ("Airflow", "Airtable"), ("Sentry", "Sentinel"),
              ("Vault", "Vercel"), ("Spark", "Sparkle"), ("Miro", "Mirage"), ("Trello", "Tribe"),
              ("Docker", "Dokku"), ("Loki", "Locust"), ("Postgres", "Prestige"), ("Argo CD", "Argo Workflows")]


def _people_ru() -> list[Entity]:
    out = []
    for i, s in enumerate(RU_SURNAMES_OV + RU_SURNAMES_OTHER):
        m = RU_MALE[i % len(RU_MALE)]
        f = RU_FEMALE[(i * 7) % len(RU_FEMALE)]
        fem = s + "а" if s.endswith(("ов", "ёв", "ев", "ин", "ын")) else s
        out.append(Entity(f"{m} {s}", "person_ru", "ru", gender="m"))
        out.append(Entity(f"{f} {fem}", "person_ru", "ru", gender="f"))
    return out


HE_FIRST = (
    "נועה דניאל תמר איתי מאיה עומר שירה יונתן רוני יעל אריאל מיכל אדם טל אלון גל אורי "
    "ליאור עידו הילה יואב עמית רותם נדב לירון מורן ענבל גיא ניר ליהי רועי דור נטע עדי "
    "אופיר מעיין שי שני עידן יובל ברק נמרוד קרן איילת דנה אבי משה יוסי רחל שרה"
).split()
HE_SURNAMES = (
    "כהן לוי מזרחי פרץ ביטון דהן אברהמי פרידמן מלכה אזולאי כץ יוסף דוד עמר אוחיון חדד "
    "גבאי בן-דוד אדרי לוין טל קליין חן שפירא חזן משה גולן ברק רוזן אבוטבול וקנין אלבז "
    "סויסה אליהו בר אילן סגל ממן שטרן גרינברג גולדשטיין וייס שוורץ ברגר הלפרין רבינוביץ "
    "אלמוג ארז פלד צור דרור הראל ינאי כרמל נאור סלע עוז שחר תמיר בראון רוזנברג"
).split()

# Hebrew phonetic transliterations of Latin tech/product terms (as produced by Hebrew ASR like zipformer2-he)
TECH_HE = {
    "ClickHouse": ["קליקהאוס", "קליק האוס"], "Kubernetes": ["קוברנטיס", "קוברנטס"], "GitHub": ["גיטהאב", "גיט האב"],
    "GitLab": ["גיטלאב", "גיט לאב"], "Slack": ["סלאק", "סלק"], "Jira": ["ג'ירה", "גירה"], "Confluence": ["קונפלואנס"],
    "Grafana": ["גרפנה", "גרפאנה"], "Prometheus": ["פרומתאוס", "פרומתיאוס"], "PostgreSQL": ["פוסטגרס", "פוסטגרסקיואל"],
    "Postgres": ["פוסטגרס"], "Redis": ["רדיס"], "Kafka": ["קפקא", "קפקה"], "Docker": ["דוקר"], "Helm": ["הלם"],
    "Terraform": ["טראפורם", "טרפורם"], "Ansible": ["אנסיבל"], "Airflow": ["איירפלו", "אירפלו"], "Figma": ["פיגמה"],
    "Notion": ["נושן", "נואושן"], "Telegram": ["טלגרם", "טלגראם"], "WhatsApp": ["וואטסאפ", "ווטסאפ"], "Zoom": ["זום"],
    "Google Meet": ["גוגל מיט"], "Python": ["פייתון", "פיתון"], "TypeScript": ["טייפסקריפט"], "JavaScript": ["ג'אווהסקריפט"],
    "React": ["ריאקט"], "Kotlin": ["קוטלין"], "Android": ["אנדרואיד"], "iOS": ["איי או אס"], "Linux": ["לינוקס"],
    "macOS": ["מק או אס"], "Windows": ["ווינדוס", "וינדוס"], "AWS": ["איי דאבל יו אס", "איי דבליו אס"],
    "Elasticsearch": ["אלסטיקסרץ'", "אלסטיק"], "MongoDB": ["מונגו", "מונגודיבי"], "MySQL": ["מאי אסקיואל"],
    "Nginx": ["אנג'ינקס", "נגינקס"], "Jenkins": ["ג'נקינס", "גנקינס"], "Sentry": ["סנטרי"], "Datadog": ["דאטאדוג", "דטאדוג"],
    "Kibana": ["קיבנה", "קיבאנה"], "Loki": ["לוקי"], "Argo CD": ["ארגו סידי"], "Istio": ["איסטיו"], "Envoy": ["אנווי"],
    "gRPC": ["ג'י אר פי סי"], "GraphQL": ["גראפקיואל"], "REST API": ["רסט איי פי איי"], "OAuth": ["או אות'"],
    "PyTorch": ["פייטורץ'", "פייטורצ'"], "TensorFlow": ["טנסורפלו"], "Hugging Face": ["האגינג פייס"],
    "ChatGPT": ["צ'אט ג'יפיטי", "צאט גיפיטי"], "Claude": ["קלוד"], "Copilot": ["קופיילוט"], "VS Code": ["וי אס קוד"],
    "IntelliJ": ["אינטליג'יי"], "Xcode": ["אקס קוד"], "Swift": ["סוויפט"], "Rust": ["ראסט"], "Golang": ["גולאנג"],
    "Java": ["ג'אווה", "גאווה"], "Django": ["ג'אנגו"], "FastAPI": ["פאסט איי פי איי"], "RabbitMQ": ["רביט אם קיו"],
    "Vault": ["וולט"], "Karpenter": ["קרפנטר"], "Bitbucket": ["ביטבאקט"], "Trello": ["טרלו"], "Miro": ["מירו"],
    "Excel": ["אקסל"], "Google Docs": ["גוגל דוקס"], "Google Sheets": ["גוגל שיטס"], "Dropbox": ["דרופבוקס"],
    "Snowflake": ["סנופלייק"], "BigQuery": ["ביגקוורי"], "Spark": ["ספארק"], "Flink": ["פלינק"],
    "Stripe": ["סטרייפ"], "PayPal": ["פייפאל"], "Netflix": ["נטפליקס"], "Spotify": ["ספוטיפיי"], "YouTube": ["יוטיוב"],
    "Instagram": ["אינסטגרם"], "LinkedIn": ["לינקדאין"], "Discord": ["דיסקורד"], "OpenAI": ["אופן איי איי"],
    "NVIDIA": ["אנבידיה", "אנווידיה"], "Samsung": ["סמסונג"], "MacBook": ["מקבוק"], "iPhone": ["אייפון"],
    "vLLM": ["וי אל אל אם"], "llama.cpp": ["לאמה סי פי פי"], "GGUF": ["ג'י ג'י יו אף"], "ONNX": ["אוניקס", "אונקס", "או אן אן אקס"],
    "Cloudflare": ["קלאודפלייר"], "Vercel": ["ורסל"], "Supabase": ["סופאבייס"], "Linear": ["ליניאר"],
    "Obsidian": ["אובסידיאן"], "Traefik": ["טרפיק"], "Parquet": ["פרקט"], "Qdrant": ["קודרנט"],
    "Gradle": ["גריידל", "גרדל"], "PR": ["פיאר", "פי אר", "פול ריקווסט"], "CI": ["סיאיי", "סי איי"],
    "API": ["אייפיאיי", "איי פי איי"], "SDK": ["אס די קיי"], "SQL": ["אס קיו אל"], "OCR": ["אוסיער", "או סי אר"],
    "ASR": ["איי אס אר"], "Fold7": ["פולד שבע", "פולד 7"], "Bit": ["ביט"], "PayBox": ["פייבוקס"],
    "Waze": ["ווייז", "וייז"], "Wolt": ["וולט"], "Monday": ["מאנדיי", "מנדיי"], "Asana": ["אסאנה"],
    "Gmail": ["ג'ימייל", "גימייל"], "Outlook": ["אאוטלוק"], "Chrome": ["כרום"], "Teams": ["טימס"],
    "Moovit": ["מוביט"], "Gett": ["גט"], "Booking": ["בוקינג"], "Airbnb": ["איירביאנבי"],
    "Todoist": ["טודואיסט"], "Cursor": ["קרסור"], "RunPod": ["ראנפוד", "רנפוד"], "Git": ["גיט"],
}

PLACES_HE = [
    "תל אביב", "ירושלים", "חיפה", "באר שבע", "הרצליה", "רמת גן", "פתח תקווה", "נתניה",
    "אשדוד", "רחובות", "כפר סבא", "רעננה", "מודיעין", "גבעתיים", "חולון", "אילת",
]


def _people_he() -> list[Entity]:
    out = []
    for i, s in enumerate(HE_SURNAMES):
        f1 = HE_FIRST[(i * 3) % len(HE_FIRST)]
        f2 = HE_FIRST[(i * 3 + 17) % len(HE_FIRST)]
        out.append(Entity(f"{f1} {s}", "person_he", "he"))
        if f2 != f1:
            out.append(Entity(f"{f2} {s}", "person_he", "he"))
    return out


def _people_en() -> list[Entity]:
    return [Entity(f"{EN_FIRST[(i * 5) % len(EN_FIRST)]} {s}", "person_en", "en") for i, s in enumerate(EN_SURNAMES)]


ALL: list[Entity] = (
    _people_ru() + _people_en() + _people_he()
    + [Entity(k, "tech", "en", tuple(v), known=k in KNOWN_TECH) for k, v in TECH.items()]
    + [Entity(c, "company", "ru" if any("а" <= ch <= "я" for ch in c.lower()) else "en") for c in COMPANIES]
    + [Entity(p, "place_ru", "ru") for p in PLACES_RU] + [Entity(p, "place_en", "en") for p in PLACES_EN]
    + [Entity(p, "place_he", "he") for p in PLACES_HE]
    + [Entity(c, "code_id", "en") for c in CODE_IDS]
)
BY_TEXT = {e.text: e for e in ALL}


def pool(kind: str | tuple, split: str) -> list[Entity]:
    kinds = (kind,) if isinstance(kind, str) else kind
    # dev scenarios may reuse train entities? No: each split draws only from its own pool.
    return [e for e in ALL if e.kind in kinds and e.split == split]


def first_name(e: Entity) -> str:
    return e.text.split()[0]


def near_miss(e: Entity, rng: random.Random, split: str, k: int = 2) -> list[Entity]:
    """Near-miss distractors from the same split: same first name/other surname,
    similar-looking product, same kind."""
    cands: list[Entity] = []
    if e.kind in ("person_ru", "person_en", "person_he"):
        fn = first_name(e)
        same_first = [x for x in pool(e.kind, split) if x is not e and first_name(x) == fn]
        sur = e.text.split()[-1]
        similar = [x for x in pool(e.kind, split) if x is not e and x.text.split()[-1][:3] == sur[:3]]
        cands = same_first + similar
        if len(cands) < k:
            cands += rng.sample([x for x in pool(e.kind, split) if x is not e], k)
    elif e.kind == "tech":
        for a, b in CONFUSABLE:
            if e.text in (a, b):
                other = b if e.text == a else a
                cands.append(BY_TEXT.get(other) or Entity(other, "tech", "en"))
        cands += rng.sample([x for x in pool("tech", split) if x is not e], k)
    else:
        cands = rng.sample([x for x in pool(e.kind, split) if x is not e] or [e], min(k, 3))
    seen, out = {e.text}, []
    for c in cands:
        if c.text not in seen:
            seen.add(c.text)
            out.append(c)
    return out[:k]
