"""Real-audio synthetic dataset pipeline for Girgur v3 (Multi-TTS -> Audio Degradation -> Multi-ASR -> 3-lang OCR).

Four resumable stages:
  1. plan      -- deterministic, leakage-free scenario plan across ru/mixed/en/he/he_en/he_ru, modern IT/AI/dev
                  terms, and diverse phone screens (Slack, Telegram+ChatGPT, GitHub, Jira, Gmail, Notion, Israeli apps).
  2. teacher   -- batched teacher-LLM generation of {screen, spoken, tts_cyrl, target} via ml.data.synth.teacher.
  3. audio     -- real multi-TTS (Supertonic3, Piper, Kokoro1, Edge-TTS Neural/Multilingual) -> probabilistic phone
                  channel degradation (clean/phone/phone_mid/far/speed) -> real multi-ASR (Parakeet v3, Parakeet Ultra,
                  GigaAM v3 RNN-T/CTC, Nemotron 3.5, T-one, Whisper, Zipformer2 HE). Run in roundtrip venv.
  4. assemble  -- real Chrome screenshot rendering + 3-lang PP-OCRv5 -> category/OCR verification -> train/dev/test
                  JSONL in exact Murmur contract format. Run in data venv with MURMUR_OCR_MODELS=.cache/ml/ocr_3lang.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from ml.common import text as T
from ml.common.render_bridge import REPO
from ml.common.util import bucket_split, rng as mkrng, stable_id

# ---------------------------------------------------------------------------
# 1. Modern IT / AI / Dev Vocabulary + Diverse Apps & Screens
# ---------------------------------------------------------------------------

# Widely known 2025-2026 AI / LLM / dev terms: model SHOULD spell these right even WITHOUT seeing them on screen.
WELL_KNOWN_IT = [
    # LLMs & AI models / coding agents
    "Codex", "Claude", "Claude Code", "Claude Opus", "Claude Sonnet", "Opus", "Sonnet", "Haiku",
    "Qwen", "Qwen3", "Qwen3.5", "GPT-4o", "GPT-5", "o3-mini", "o4-mini", "Gemini", "Gemini 2.5",
    "DeepSeek", "DeepSeek-R1", "Llama 4", "Mistral", "Grok", "Perplexity", "ChatGPT", "Anthropic", "OpenAI",
    "Whisper", "ElevenLabs", "Midjourney", "Sora", "Cursor", "Windsurf", "Devin", "GitHub Copilot",
    # ML / LLM engineering
    "LoRA", "QLoRA", "GGUF", "ONNX", "vLLM", "llama.cpp", "Ollama", "MCP", "RAG", "SFT", "RLHF", "DPO", "GRPO",
    "PyTorch", "Hugging Face", "TensorRT", "Triton", "CUDA", "FlashAttention", "BFloat16", "FP8", "Q4_0", "INT8",
    "RunPod", "Modal", "WandB", "LangChain", "LangGraph", "ComfyUI", "RTX 5090", "A100", "H100",
    # Backend / Frontend / Infra / Mobile
    "FastAPI", "Pydantic", "Supabase", "Vercel", "Cloudflare", "Stripe", "Linear", "Sentry", "Datadog",
    "ClickHouse", "DuckDB", "Polars", "Postgres", "PostgreSQL", "Redis", "Kafka", "Docker", "Kubernetes",
    "GitHub", "GitLab", "Slack", "Jira", "Notion", "Obsidian", "Figma", "Postman", "Grafana", "Nginx", "Terraform",
    "Next.js", "React", "TypeScript", "Python", "Kotlin", "Rust", "Golang", "Tailwind", "Vite", "Bun",
    "uv", "ruff", "pytest", "Gradle", "Cargo", "Helm", "ArgoCD", "Prometheus", "OpenTelemetry", "Playwright",
    "Prisma", "Drizzle", "tRPC", "GraphQL", "gRPC", "WebSocket", "WebRTC", "Tailscale", "WireGuard",
    "Jetpack Compose", "Flutter", "Ktor", "OkHttp", "Protobuf", "AWS", "S3", "GCP", "BigQuery", "Vertex AI",
    "CI/CD", "PR", "OOM", "API", "SDK", "CLI", "SSH", "JWT", "OAuth", "VRAM", "GPU",
]

# Custom / project / newer / niche tools: screen OCR context SHOULD disambiguate them!
# Hash-split across train/dev/test so dev/test test genuine unseen custom names from screen!
CUSTOM_IT = [
    "OpenClaw", "Antigravity", "Murmur", "Girgur", "Hermes", "Parakeet", "Zipformer", "Zipformer2",
    "GigaAM", "Nemotron", "Supertonic", "Kokoro", "sherpa-onnx", "Silero", "PaddleOCR", "PP-OCRv5",
    "Chatterbox", "Piper", "Kaldi", "SM8750", "Fold7", "QNN", "MooseFS",
    "Aider", "Cline", "Roo Code", "Zed", "Ghostty", "WezTerm", "Raycast", "Logseq", "n8n",
    "Temporal", "Prefect", "Airflow", "Dagster", "PostHog", "Plausible", "Coolify", "Dokploy",
    "Fly.io", "Railway", "Neon", "Turso", "Upstash", "Meilisearch", "Typesense", "Qdrant",
    "Milvus", "Weaviate", "ChromaDB", "Pinecone", "LiteLLM", "OpenRouter", "Together", "Groq",
    "Cerebras", "Fireworks", "SambaNova", "Bolt.new", "Lovable", "Replit", "Tabnine", "Continue",
    "Keycloak", "Authentik", "Traefik", "Caddy", "Cilium", "Istio", "Envoy", "Karpenter",
    "ClickUp", "Coda", "Airtable", "Retool", "Metabase", "Superset", "Trino", "StarRocks",
    "Vitest", "Biome", "oxlint", "Turborepo", "Nx", "Esbuild", "SWC", "Ruff", "Mypy", "Pyright",
    "vLLM-Omni", "SGLang", "ExLlamaV2", "MLX", "Unsloth", "Axolotl", "TRL", "PEFT", "DeepSpeed",
    "FSDP", "Megatron", "Ray", "SkyPilot", "BentoML", "KServe", "Modal Labs", "Baseten",
]

# Everyday Russian IT loanwords (must stay clean Russian Cyrillic in RU/mixed targets)
RU_DEV_JARGON = [
    "бэкенд", "фронтенд", "фулстек", "деплой", "задеплоить", "пул-реквест", "коммит", "мердж",
    "рефакторинг", "микросервис", "эндпоинт", "апишка", "кэш", "токен", "промпт", "эмбеддинг",
    "файнтюн", "квантизация", "инференс", "пайплайн", "прод", "стейджинг", "докер", "кубер",
    "багфикс", "хотфикс", "релиз", "репозиторий", "ветка", "миграция", "вебхук", "контейнер",
    "воркер", "очередь", "датасет", "бенчмарк", "латенси", "галлюцинация", "адаптер", "контекст",
]

# Everyday Hebrew IT loanwords
HE_DEV_JARGON = [
    "בקאנד", "פרונטאנד", "דיפלוי", "פול-רקווסט", "קומיט", "מרג׳", "ריפקטורינג", "מיקרוסרוויס",
    "אנדפוינט", "קאש", "טוקן", "פרומפט", "אמבדינג", "פיין-טיונינג", "קוונטיזציה", "אינפרנס",
    "פייפליין", "פרודקשן", "סטייג׳ינג", "דוקר", "קוברנטיס", "באגפיקס", "הוטפיקס", "ריליס",
    "סקריפט", "ריפו", "בראנץ׳", "מיגרציה", "וובהוק", "קונטיינר", "וורקר", "דאטהסט", "בנצ׳מרק",
]

# Confusable / near-miss pairs (screen shows distractor, user says target!)
NEAR_MISS_PAIRS = [
    ("Codex", "Codec"), ("OpenClaw", "OpenAI"), ("OpenClaw", "OpenRouter"), ("Claude", "Cloudflare"),
    ("Qwen3.5", "Qwen3"), ("GPT-5", "GPT-4o"), ("ClickHouse", "ClickUp"), ("FastAPI", "Fastify"),
    ("Cursor", "Caddy"), ("LoRA", "Lora"), ("vLLM", "VLM"), ("GGUF", "GGML"), ("Gradle", "Grafana"),
    ("Parakeet", "Parquet"), ("Zipformer2", "Transformer"), ("Nemotron", "NeMo"), ("DeepSeek-R1", "DeepSpeed"),
    ("Sentry", "Svelte"), ("Prisma", "Prometheus"), ("Qdrant", "Quadrant"), ("Supabase", "Superbase"),
    ("Linear", "Linux"), ("Docker", "Dokploy"), ("GitHub", "GitLab"), ("Opus", "OpenSSL"),
    ("Sonnet", "Socket"), ("Gemini 2.5", "Gemini 2.0"), ("RunPod", "Podman"), ("Hermes", "Helm"),
    ("Murmur", "Mermaid"), ("Antigravity", "Ansible"), ("Redis", "Redux"), ("Kafka", "Karpenter"),
    ("Tailscale", "Tailwind"), ("Datadog", "Dagster"), ("Playwright", "Pyright"), ("SGLang", "Golang"),
]

# Diverse phone app contexts (all packages exist in prompts/tones/app_categories.v1.json or default!)
TECH_APPS = [
    {"package": "com.Slack", "label": "Slack", "category": "work_chat", "field": "message",
     "scene_hint": "Slack work channel (#backend, #ml-infra, #releases, #incidents, #ai-agents, #mobile-dev)"},
    {"package": "org.telegram.messenger", "label": "Telegram", "category": "messaging", "field": "message",
     "scene_hint": "Telegram chat with ChatGPT bot or a developer colleague discussing backend/AI architecture"},
    {"package": "org.telegram.messenger", "label": "Telegram", "category": "messaging", "field": "message",
     "scene_hint": "Telegram work group chat (Backend Team / ML Ops / Infra Alerts)"},
    {"package": "com.openai.chatgpt", "label": "ChatGPT", "category": "default", "field": "generic",
     "scene_hint": "ChatGPT mobile chat debugging backend/frontend code, SQL, or LLM prompts"},
    {"package": "com.anthropic.claude", "label": "Claude", "category": "default", "field": "generic",
     "scene_hint": "Claude mobile chat discussing architecture, LoRA training, or PR review"},
    {"package": "com.google.android.apps.bard", "label": "Gemini", "category": "default", "field": "generic",
     "scene_hint": "Gemini mobile chat comparing models, infra, or writing code"},
    {"package": "com.github.android", "label": "GitHub", "category": "code", "field": "code",
     "scene_hint": "GitHub pull request review comment or issue discussion"},
    {"package": "com.atlassian.android.jira.core", "label": "Jira", "category": "work_chat", "field": "message",
     "scene_hint": "Jira sprint ticket comment about a backend/frontend bug or deployment"},
    {"package": "com.linear.app", "label": "Linear", "category": "work_chat", "field": "message",
     "scene_hint": "Linear engineering issue tracking an API bug, latency spike, or model eval"},
    {"package": "com.monday.monday", "label": "Monday.com", "category": "work_chat", "field": "message",
     "scene_hint": "Monday.com R&D board item update"},
    {"package": "com.discord", "label": "Discord", "category": "work_chat", "field": "message",
     "scene_hint": "Discord developer community channel about local LLMs, GGUF, or coding agents"},
    {"package": "com.google.android.gm", "label": "Gmail", "category": "email", "field": "email_body",
     "scene_hint": "Gmail engineering email thread with colleagues or vendors"},
    {"package": "notion.id", "label": "Notion", "category": "notes", "field": "note",
     "scene_hint": "Notion technical RFC / architecture notes / sprint retro"},
    {"package": "md.obsidian", "label": "Obsidian", "category": "notes", "field": "note",
     "scene_hint": "Obsidian developer log / experiment notes"},
    {"package": "com.whatsapp", "label": "WhatsApp", "category": "messaging", "field": "message",
     "scene_hint": "WhatsApp quick chat with a tech lead or co-founder"},
    {"package": "com.android.chrome", "label": "Chrome", "category": "search", "field": "search",
     "scene_hint": "Chrome search query or documentation page"},
]

EVERYDAY_APPS = [
    {"package": "com.whatsapp", "label": "WhatsApp", "category": "messaging", "field": "message",
     "scene_hint": "WhatsApp personal or family chat"},
    {"package": "org.telegram.messenger", "label": "Telegram", "category": "messaging", "field": "message",
     "scene_hint": "Telegram personal chat or neighborhood group"},
    {"package": "com.google.android.gm", "label": "Gmail", "category": "email", "field": "email_body",
     "scene_hint": "Gmail personal or administrative email"},
    {"package": "com.microsoft.office.outlook", "label": "Outlook", "category": "email", "field": "email_body",
     "scene_hint": "Outlook work or client email"},
    {"package": "com.samsung.android.app.notes", "label": "Samsung Notes", "category": "notes", "field": "note",
     "scene_hint": "Quick personal note or checklist"},
    {"package": "com.google.android.keep", "label": "Keep Notes", "category": "notes", "field": "note",
     "scene_hint": "Shopping list or quick memo"},
    {"package": "com.google.android.calendar", "label": "Calendar", "category": "default", "field": "generic",
     "scene_hint": "Calendar event or reminder"},
    {"package": "com.waze", "label": "Waze", "category": "search", "field": "search",
     "scene_hint": "Waze navigation address search"},
    {"package": "com.wolt.android", "label": "Wolt", "category": "default", "field": "generic",
     "scene_hint": "Wolt courier delivery instructions"},
    {"package": "com.bnhp.payments.app", "label": "Bit", "category": "default", "field": "generic",
     "scene_hint": "Bit money transfer note"},
    {"package": "com.maccabi.health", "label": "Maccabi", "category": "default", "field": "generic",
     "scene_hint": "Maccabi clinic appointment or doctor message"},
    {"package": "com.android.chrome", "label": "Chrome", "category": "search", "field": "search",
     "scene_hint": "Chrome web search"},
]

ADDRESS_APPS = [
    {"package": "com.google.android.gm", "label": "Gmail", "category": "email", "field": "address",
     "scene_hint": "Gmail To: email address field"},
    {"package": "com.microsoft.office.outlook", "label": "Outlook", "category": "email", "field": "address",
     "scene_hint": "Outlook recipient email field"},
    {"package": "com.android.chrome", "label": "Chrome", "category": "search", "field": "address",
     "scene_hint": "Chrome URL address bar"},
]

IT_TOPICS = [
    "debugging a backend API 500 error in production",
    "reviewing a frontend React/TypeScript PR and state bug",
    "comparing LLM coding agents and model eval metrics",
    "fine-tuning a LoRA adapter and exporting Q4_0 GGUF",
    "fixing a slow ClickHouse or Postgres query",
    "setting up an MCP server and tool-calling workflow",
    "investigating an OOM crash on a GPU training pod",
    "asking ChatGPT in Telegram to refactor an async backend endpoint",
    "triaging a CI/CD pipeline failure and flaky Gradle tests",
    "migrating a microservice to Docker and Kubernetes",
    "discussing voice ASR latency and streaming transcription bugs",
    "configuring Cloudflare Workers, caching, and webhooks",
    "planning a sprint release and hotfix rollback",
    "benchmarking vLLM / llama.cpp throughput and quantization",
    "writing a technical RFC for a RAG and embeddings pipeline",
    "fixing mobile app UI layout, RTL text, and OCR errors",
]

EVERYDAY_TOPICS = [
    "running 15 minutes late to a meeting",
    "coordinating a street address and entrance code for delivery",
    "booking a dentist or clinic appointment",
    "splitting a dinner bill or rent transfer",
    "planning a weekend trip, flights, and hotel",
    "picking up kids from school or kindergarten",
    "asking a friend for a restaurant recommendation",
    "confirming a repair technician or courier arrival time",
    "making a grocery and pharmacy checklist",
    "discussing an apartment lease, Arnona, or Vaad Bayit",
    "sending an email address or link to a colleague",
    "summarizing action items after a call",
]

HE_HARD_TOPICS = [
    "Israeli street address with house number and floor (Rothschild 45, Dizengoff 120, Herzl 18, Ben Yehuda 72)",
    "spoken Hebrew email address (name dot surname shtrudel gmail dot com / walla dot co dot il)",
    "starting with Hebrew filler 'טוב אז' or 'אממ טוב אז' before a real work or everyday message",
    "Hebrew+English tech update with PR/ticket numbers (#814, v3.5) and ב-Gradle / ב-Slack / בבקאנד / בפרונטאנד",
    "Israeli everyday Russian+Hebrew bilingual message (теудат зеут, битуах леуми, купат холим, арнона, мазган)",
    "Wolt / Bit / Maccabi / Waze message with Israeli phone number (050/052/054) and amount in ₪",
]

# Archetype distribution (sums to 100)
ARCHETYPES = [
    ("it_with_screen", 24),           # modern IT/AI/dev terms visible on Slack/Telegram/GitHub/ChatGPT screen
    ("it_no_screen_term", 18),        # WELL_KNOWN IT/AI/dev terms NOT on screen (teaches semantic homophone repair!)
    ("it_custom_screen", 14),         # CUSTOM/niche terms (OpenClaw, Antigravity, Murmur, Parakeet...) visible on screen
    ("near_miss", 8),                 # look-alike term on screen that was NOT said
    ("self_correction", 9),           # spoken self-correction with IT or everyday content
    ("hebrew_hard", 7),               # targeted Israeli Hebrew hard cases (only for he/he_en/he_ru)
    ("numbers_and_lists", 6),         # versions, prices, times, numbered lists
    ("address", 4),                   # spoken email / URL across RU/EN/HE
    ("dictated_question_or_prompt", 5),  # questions or ChatGPT/Claude prompts (transcribe verbatim, never answer!)
    ("everyday_diverse", 3),          # non-tech everyday life across diverse apps
    ("no_op_and_clean", 2),           # clean dictation that needs minimal/no edits
]

LANG_WEIGHTS = [
    ("ru", 25),
    ("mixed", 23),
    ("en", 20),
    ("he", 14),
    ("he_en", 11),
    ("he_ru", 7),
]


def _pick_w(r: random.Random, items):
    vals, ws = zip(*items)
    return r.choices(vals, weights=ws, k=1)[0]


def _custom_pool(split: str) -> list[str]:
    return [t for t in CUSTOM_IT if bucket_split("v3term:" + t, dev=0.10, test=0.10) == split]


def _near_miss_pool(split: str) -> list[tuple[str, str]]:
    return [p for p in NEAR_MISS_PAIRS if bucket_split("v3nm:" + p[0], dev=0.10, test=0.10) == split]


def make_plan(count: int, seed: int = 20261125) -> list[dict]:
    """Generate deterministic, leakage-free scenarios split ~86% train / 7% dev / 7% test."""
    n_dev = max(4, round(count * 0.07))
    n_test = max(4, round(count * 0.07))
    n_train = count - n_dev - n_test
    splits = [("train", n_train), ("dev", n_dev), ("test", n_test)]
    scenarios = []
    idx = 0
    for split, n in splits:
        custom = _custom_pool(split) or CUSTOM_IT[:10]
        nm_pool = _near_miss_pool(split) or NEAR_MISS_PAIRS[:6]
        for j in range(n):
            idx += 1
            r = mkrng("audio_v3_plan", seed, split, j)
            arch = _pick_w(r, ARCHETYPES)
            if arch == "hebrew_hard":
                lang = _pick_w(r, [("he", 45), ("he_en", 35), ("he_ru", 20)])
            else:
                lang = _pick_w(r, LANG_WEIGHTS)

            # Pick app & screen mode
            if arch == "address":
                app = dict(r.choice(ADDRESS_APPS))
                screen_mode = "empty" if r.random() < 0.4 else "full"
            elif arch in ("it_with_screen", "it_custom_screen", "near_miss"):
                app = dict(r.choice(TECH_APPS))
                screen_mode = "full"
            elif arch == "it_no_screen_term":
                app = dict(r.choice(TECH_APPS))
                screen_mode = "absent" if r.random() < 0.30 else "unrelated"
            elif arch == "dictated_question_or_prompt":
                app = dict(r.choice([a for a in TECH_APPS if a["label"] in ("ChatGPT", "Claude", "Gemini", "Telegram", "Slack", "Chrome")]))
                screen_mode = "full" if r.random() < 0.7 else "empty"
            elif arch == "everyday_diverse":
                app = dict(r.choice(EVERYDAY_APPS))
                screen_mode = "absent" if r.random() < 0.25 else "full"
            else:
                app = dict(r.choice(TECH_APPS if r.random() < 0.65 else EVERYDAY_APPS))
                screen_mode = "absent" if r.random() < 0.2 else "full"

            # Pick terms
            said_terms: list[str] = []
            screen_must_show: list[str] = []
            must_not_show: list[str] = []
            jargon: list[str] = []

            if arch == "it_with_screen":
                k_wk = r.choice([1, 2, 2])
                k_cu = 1 if r.random() < 0.45 else 0
                said_terms = r.sample(WELL_KNOWN_IT, k=k_wk) + (r.sample(custom, k=1) if k_cu else [])
                screen_must_show = list(said_terms)
                if lang in ("ru", "mixed"):
                    jargon = r.sample(RU_DEV_JARGON, k=r.choice([1, 2]))
                elif lang in ("he", "he_en"):
                    jargon = r.sample(HE_DEV_JARGON, k=r.choice([1, 2]))
            elif arch == "it_no_screen_term":
                said_terms = r.sample(WELL_KNOWN_IT, k=r.choice([1, 2, 2]))
                must_not_show = list(said_terms)
                if lang in ("ru", "mixed"):
                    jargon = r.sample(RU_DEV_JARGON, k=r.choice([1, 2]))
                elif lang in ("he", "he_en"):
                    jargon = r.sample(HE_DEV_JARGON, k=r.choice([1, 2]))
            elif arch == "it_custom_screen":
                c_term = r.choice(custom)
                extra = r.sample(WELL_KNOWN_IT, k=1) if r.random() < 0.6 else []
                said_terms = [c_term] + extra
                screen_must_show = [c_term] + (extra if r.random() < 0.5 else [])
                if lang in ("ru", "mixed"):
                    jargon = r.sample(RU_DEV_JARGON, k=1)
                elif lang in ("he", "he_en"):
                    jargon = r.sample(HE_DEV_JARGON, k=1)
            elif arch == "near_miss":
                t_said, t_dist = r.choice(nm_pool)
                said_terms = [t_said]
                screen_must_show = [t_dist]
                must_not_show = [t_said]
            elif arch in ("self_correction", "dictated_question_or_prompt", "numbers_and_lists"):
                if r.random() < 0.7:
                    said_terms = r.sample(WELL_KNOWN_IT, k=r.choice([1, 2]))
                    if screen_mode == "full" and r.random() < 0.5:
                        screen_must_show = [said_terms[0]]
                    if lang in ("ru", "mixed"):
                        jargon = r.sample(RU_DEV_JARGON, k=1)
            elif arch == "hebrew_hard":
                if lang == "he_en":
                    said_terms = r.sample(WELL_KNOWN_IT, k=r.choice([1, 2]))
                    if r.random() < 0.5:
                        screen_must_show = [said_terms[0]]
                    jargon = r.sample(HE_DEV_JARGON, k=1)

            topic = (r.choice(HE_HARD_TOPICS) if arch == "hebrew_hard"
                     else r.choice(IT_TOPICS) if arch.startswith("it_") or arch in ("near_miss", "dictated_question_or_prompt")
                     else r.choice(IT_TOPICS if r.random() < 0.5 else EVERYDAY_TOPICS))
            length_bucket = _pick_w(r, [((4, 12), 30), ((13, 28), 48), ((29, 55), 22)])
            if arch == "address":
                length_bucket = (4, 12)

            sid = f"av3-{split[:2]}-{j:05d}"
            scenarios.append({
                "sid": sid,
                "split": split,
                "archetype": arch,
                "lang": lang,
                "app": {"package": app["package"], "label": app["label"], "category": app["category"]},
                "field_kind": app["field"],
                "scene_hint": app["scene_hint"],
                "screen_mode": screen_mode,
                "topic": topic,
                "said_terms": said_terms,
                "jargon": jargon,
                "screen_must_show": screen_must_show,
                "must_not_show": must_not_show,
                "length": list(length_bucket),
                "register": r.choice(["casual", "neutral", "professional"]),
                "seed": r.randrange(1 << 30),
            })
    return scenarios


# ---------------------------------------------------------------------------
# 2. Teacher Prompt & Batch Generation
# ---------------------------------------------------------------------------

TEACHER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["examples"],
    "properties": {
        "examples": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sid", "screen", "spoken", "tts_cyrl", "target"],
                "properties": {
                    "sid": {"type": "string", "maxLength": 40},
                    "screen": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "messages", "before_cursor", "snippets"],
                        "properties": {
                            "title": {"type": "string", "maxLength": 120},
                            "messages": {
                                "type": "array",
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["author", "is_me", "text"],
                                    "properties": {
                                        "author": {"type": "string", "maxLength": 60},
                                        "is_me": {"type": "boolean"},
                                        "text": {"type": "string", "maxLength": 360},
                                    },
                                },
                            },
                            "before_cursor": {"type": "string", "maxLength": 200},
                            "snippets": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 160}},
                        },
                    },
                    "spoken": {"type": "string", "maxLength": 1200},
                    "tts_cyrl": {"type": "string", "maxLength": 1200},
                    "target": {"type": "string", "maxLength": 1200},
                },
            },
        }
    },
}

TEACHER_SYSTEM = """You generate realistic phone-dictation training scenarios whose "spoken" text will be synthesized by REAL TTS engines and transcribed by REAL ASR models (Parakeet v3, GigaAM v3, Nemotron 3.5, Zipformer2 Hebrew).

Return {"examples": [...]} in exact order, copying each scenario's "sid".

Fields per example:
1. "screen": realistic phone screen visible while the user dictates (rendered to PNG and read by OCR).
   - "title": channel name (e.g. "#backend-alerts", "ChatGPT", "Backend Team", "PR #814: Fix LoRA loader", "Алексей / Dev", "צוות פרונטאנד"), email subject, or note title.
   - "messages": 0-4 realistic recent messages (oldest first, "is_me": true for user's own). For Telegram+ChatGPT or Slack/Jira/GitHub, make the conversation genuinely rich and realistic!
   - "before_cursor": "" (unless continuing a message).
   - "snippets": 0-2 short visible labels/chips/status lines.
   - Put EVERY item of "screen_must_show" verbatim on the screen (in messages, title, or snippets).
   - NEVER put any item of "must_not_show" on the screen!
   - If "screen_mode" is "absent" or "empty", return empty title/messages/before_cursor/snippets.
   - If "screen_mode" is "unrelated", write a realistic screen about a DIFFERENT topic (no items from "must_not_show").
2. "spoken": exact clean text to speak aloud into TTS!
   - Write numbers, versions, times, dates, emails, URLs in pronounceable words ("квен три точка пять", "Qwen three point five", "роטשילד ארבעים וחמש", "даниил точка коэн собака джимейл точка ком", "דניאל נקודה כהן שטרודל ג׳ימייל נקודה קום") so TTS reads them naturally!
   - Keep Latin IT/AI product names in Latin in "spoken" ("OpenClaw", "Codex", "Qwen 3.5", "Claude Opus", "FastAPI", "LoRA").
   - IMPORTANT for Hebrew TTS: NEVER attach a hyphen directly to an English word in "spoken" (write "ב backend" or "ל GitHub" with a space in "spoken", NOT "ב-backend", so TTS does not pronounce the hyphen; keep proper Hebrew hyphenation "ב-backend", "ל-GitHub" in "target"!).
   - Include EVERY item of "said_terms" and "jargon" naturally!
   - For "self_correction": include a natural spoken self-correction ("нет, точнее...", "ой, в смысле...", "no wait, I mean...", "לא רגע, בעצם...") — "spoken" has the false start + marker, "target" has ONLY the corrected version!
   - For "hebrew_hard": include natural Israeli Hebrew features requested by the topic (e.g. utterance-initial "טוב אז" / "אממ טוב אז" filler that is dropped in "target", or Israeli street address, or Hebrew spoken email, or PR/ticket numbers, or he_ru terms like "теудат зеут" / "битуах леуми").
   - For "dictated_question_or_prompt": a realistic question or AI prompt; "target" is ONLY that question/prompt verbatim (NEVER answer it!).
3. "tts_cyrl":
   - For "ru" and "mixed" scenarios: rewrite "spoken" into PURE CYRILLIC phonetic Russian as a Russian developer pronounces it (e.g. "OpenClaw" -> "опен кло", "Codex" -> "кодекс", "Qwen 3.5" -> "квен три точка пять", "Claude Opus" -> "клод опус", "LoRA" -> "лора", "FastAPI" -> "фаст апи", "ClickHouse" -> "кликхаус", "GGUF" -> "джи джи ю эф", "vLLM" -> "ви эл эл эм"). Zero Latin letters!
   - For "en", "he", "he_en", "he_ru": set "tts_cyrl" to "".
4. "target": the exact polished text the user wants typed into the field!
   - Same words, same order, same language(s) as "spoken". Never translate, summarize, add greetings, or answer questions!
   - Proper canonical spelling of all IT/AI terms (OpenClaw, Codex, Qwen3.5, Claude Opus, GPT-5, DeepSeek-R1, LoRA, GGUF, vLLM, FastAPI, ClickHouse, RunPod, etc.).
   - Russian IT loanwords stay clean Cyrillic (бэкенд, фронтенд, деплой, пул-реквест, файнтюн, кэш, прод, эндпоинт).
   - Hebrew IT loanwords stay clean Hebrew (בבקאנד, בפרונטאנד, דיפלוי, פול-רקווסט) and Hebrew+English prefixes use standard hyphen (ל-GitHub, ב-Slack, ב-Gradle, מ-Docker, ה-PR, על Qwen3.5).
   - Numbers/versions/prices/addresses/times formatted cleanly with digits (רוטשילד 45, PR #814, 15:30, 250 ₪, v3.5, 25%).
   - For "address": ONLY the bare lowercase email or URL (no trailing period, no spaces).
   - For "search" field: search query style (no trailing period).
   - Russian: proper ё where appropriate. Natural modern colloquial developer & everyday speech — vary openings and sentence structures wildly across the batch!

Output ONLY the JSON object."""


def run_teacher(out_dir: Path, batch_size: int = 8, concurrency: int = 6, max_requests: int = 0) -> None:
    from ml.data.synth.teacher import OpenAITeacher, TeacherConfig, load_results, run_sharded

    scenarios = [json.loads(l) for l in (out_dir / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    shards_dir = out_dir / "teacher_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    done = load_results(shards_dir, "gen")

    jobs = []
    for i in range(0, len(scenarios), batch_size):
        chunk = scenarios[i : i + batch_size]
        jid = f"b{i // batch_size:05d}"
        payload = []
        for s in chunk:
            item = {
                "sid": s["sid"],
                "language": s["lang"],
                "archetype": s["archetype"],
                "app": s["app"]["label"],
                "field": s["field_kind"],
                "scene_hint": s["scene_hint"],
                "screen_mode": s["screen_mode"],
                "topic": s["topic"],
                "register": s["register"],
                "spoken_words": f"{s['length'][0]}-{s['length'][1]}",
            }
            if s["said_terms"]:
                item["said_terms"] = s["said_terms"]
            if s["jargon"]:
                item["jargon"] = s["jargon"]
            if s["screen_must_show"]:
                item["screen_must_show"] = s["screen_must_show"]
            if s["must_not_show"]:
                item["must_not_show"] = s["must_not_show"]
            payload.append(item)
        jobs.append({
            "id": jid,
            "instructions": TEACHER_SYSTEM,
            "input": json.dumps({"scenarios": payload}, ensure_ascii=False),
            "schema": TEACHER_SCHEMA,
            "name": "audio_v3_batch",
            "gen": "audio_v3.1",
        })

    todo_n = sum(1 for j in jobs if j["id"] not in done)
    print(f"[teacher] total scenarios={len(scenarios)} batches_done={len(done)} batches_todo={todo_n}", flush=True)
    if todo_n == 0:
        return
    cfg = TeacherConfig(concurrency=concurrency, max_requests=max_requests, max_output_tokens=10000)
    teacher = OpenAITeacher(cfg)
    st = run_sharded(jobs, teacher, shards_dir, "gen", shard_size=25, concurrency=concurrency)
    print(f"[teacher] finished: {json.dumps(st)}", flush=True)


# ---------------------------------------------------------------------------
# 3. Real Multi-TTS -> Audio Degradation -> Real Multi-ASR (run in roundtrip venv!)
# ---------------------------------------------------------------------------

EDGE_HE_VOICES = ["he-IL-AvriNeural", "he-IL-HilaNeural"]
EDGE_MULTI_VOICES = [
    "en-US-AndrewMultilingualNeural", "en-US-AvaMultilingualNeural",
    "en-US-BrianMultilingualNeural", "en-US-EmmaMultilingualNeural",
]
EDGE_EN_VOICES = [
    "en-US-GuyNeural", "en-US-AndrewNeural", "en-US-AvaNeural",
    "en-US-EmmaNeural", "en-US-BrianNeural", "en-GB-RyanNeural",
]
PIPER_RU_VOICES = ["ru_RU-denis-medium", "ru_RU-dmitri-medium"]
PIPER_EN_VOICES = [
    "en_US-joe-medium", "en_US-john-medium", "en_US-kristin-medium", "en_US-norman-medium",
    "en_GB-alba-medium", "en_GB-cori-medium", "en_US-libritts_r-medium#0", "en_US-libritts_r-medium#263",
]
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nova", "af_sarah", "af_sky", "af_jessica",
    "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_puck",
    "bf_alice", "bf_emma", "bm_daniel", "bm_george", "bm_lewis",
]
SUPERTONIC_VOICES = [f"sid{i}" for i in range(10)]

SPOKEN_FILLERS = {
    "ru": ["ну", "э-э", "типа", "короче", "вот", "эм"],
    "mixed": ["ну", "э-э", "типа", "короче", "вот"],
    "en": ["um", "uh", "like", "you know", "so uh"],
    "he": ["אממ", "כאילו", "טוב אז", "אז אממ", "אה"],
    "he_en": ["אממ", "כאילו", "טוב אז", "אה"],
    "he_ru": ["אממ", "ну", "טוב אז", "э-э"],
}


def _inject_spoken_filler(text: str, lang: str, r: random.Random) -> tuple[str, bool]:
    """Naturally prepend or insert a spoken filler before TTS (~25% chance)."""
    if not text or r.random() > 0.28:
        return text, False
    f = r.choice(SPOKEN_FILLERS.get(lang, ["um"]))
    if r.random() < 0.65:
        return f"{f}, {text}", True
    words = text.split()
    if len(words) >= 6:
        pos = r.randint(2, min(5, len(words) - 2))
        words.insert(pos, f + ",")
        return " ".join(words), True
    return f"{f}, {text}", True


def _clean_for_he_tts(text: str) -> str:
    """Prevent Hebrew Edge-TTS from spelling out hyphens between Hebrew prefixes and Latin words."""
    # ב-backend -> ב backend
    text = re.sub(r"([\u05d0-\u05ea])-([A-Za-z0-9])", r"\1 \2", text)
    return text


def _choose_tts_plan(lang: str, has_cyrl: bool, r: random.Random) -> tuple[str, str, str, str]:
    """Return (engine, voice, rate, pitch)."""
    rate = r.choice(["-8%", "+0%", "+0%", "+8%", "+14%"])
    pitch = r.choice(["-2Hz", "+0Hz", "+0Hz", "+2Hz"])
    if lang in ("ru", "mixed"):
        choices = [("supertonic3", 42), ("edge_multi", 30)]
        if has_cyrl:
            choices.append(("piper_ru", 28))
        eng = _pick_w(r, choices)
        if eng == "supertonic3":
            return "supertonic3", r.choice(SUPERTONIC_VOICES), "+0%", "+0Hz"
        if eng == "piper_ru":
            return "piper", r.choice(PIPER_RU_VOICES), "+0%", "+0Hz"
        return "edge", r.choice(EDGE_MULTI_VOICES), rate, pitch
    if lang == "en":
        eng = _pick_w(r, [("kokoro1", 38), ("piper_en", 24), ("supertonic3", 16), ("edge_en", 22)])
        if eng == "kokoro1":
            return "kokoro1", r.choice(KOKORO_VOICES), "+0%", "+0Hz"
        if eng == "piper_en":
            return "piper", r.choice(PIPER_EN_VOICES), "+0%", "+0Hz"
        if eng == "supertonic3":
            return "supertonic3", r.choice(SUPERTONIC_VOICES), "+0%", "+0Hz"
        return "edge", r.choice(EDGE_EN_VOICES), rate, pitch
    # Hebrew / he_en / he_ru
    if lang == "he_ru":
        eng = _pick_w(r, [("edge_multi", 65), ("edge_he", 35)])
    elif lang == "he_en":
        eng = _pick_w(r, [("edge_he", 75), ("edge_multi", 25)])
    else:
        eng = "edge_he"
    v = r.choice(EDGE_HE_VOICES if eng == "edge_he" else EDGE_MULTI_VOICES)
    return "edge", v, rate, pitch


def _choose_asr_model(lang: str, r: random.Random) -> str:
    if lang == "ru":
        return _pick_w(r, [
            ("parakeet-tdt-0.6b-v3@onnx-int8", 48),
            ("gigaam-v3-e2e-rnnt@onnx-int8", 22),
            ("gigaam-v3-e2e-ctc@onnx-int8", 12),
            ("nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms", 10),
            ("t-one@onnx", 8),
        ])
    if lang == "mixed":
        return _pick_w(r, [
            ("parakeet-tdt-0.6b-v3@onnx-int8", 58),
            ("parakeet-ultra-0.6b@onnx-int8", 16),
            ("nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms", 14),
            ("gigaam-v3-e2e-rnnt@onnx-int8", 12),
        ])
    if lang == "en":
        return _pick_w(r, [
            ("parakeet-tdt-0.6b-v3@onnx-int8", 50),
            ("parakeet-ultra-0.6b@onnx-int8", 18),
            ("nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms", 18),
            ("whisper-small@onnx-int8", 14),
        ])
    if lang == "he_ru":
        return _pick_w(r, [
            ("zipformer2-66m-streaming-he@onnx-int8", 65),
            ("whisper-large-v3-turbo@onnx-int8", 35),
        ])
    # he / he_en
    return _pick_w(r, [
        ("zipformer2-66m-streaming-he@onnx-int8", 88),
        ("whisper-large-v3-turbo@onnx-int8", 12),
    ])


def _choose_channel(r: random.Random) -> tuple[str, float]:
    """Return (channel_preset, speed_factor)."""
    preset = _pick_w(r, [("clean", 35), ("phone", 40), ("phone_mid", 18), ("far", 7)])
    speed = 1.0
    if r.random() < 0.12:
        speed = r.choice([0.94, 0.97, 1.05, 1.08])
    return preset, speed


# Rough cross-script phonetic folding for quality gating (so Codex ~ кодекс ~ קודקס!)
_CYR_TO_LAT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "z", "з": "z",
    "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "c", "ш": "s", "щ": "s",
    "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "u", "я": "a",
})
_HEB_TO_LAT = str.maketrans({
    "א": "a", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "v", "ז": "z", "ח": "h", "ט": "t",
    "י": "i", "כ": "k", "ך": "k", "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n", "ס": "s",
    "ע": "a", "פ": "p", "ף": "p", "צ": "c", "ץ": "c", "ק": "k", "ר": "r", "ש": "s", "ת": "t",
})


def _phon_fold(s: str) -> str:
    s = s.lower().translate(_CYR_TO_LAT).translate(_HEB_TO_LAT)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _script_shares(s: str) -> tuple[float, float, float]:
    letters = [c for c in s if c.isalpha()]
    n = max(1, len(letters))
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff") / n
    heb = sum(1 for c in letters if "\u0590" <= c <= "\u05ff") / n
    lat = sum(1 for c in letters if ("A" <= c <= "Z") or ("a" <= c <= "z")) / n
    return cyr, heb, lat


def _asr_plausible(hyp: str, spoken: str, target: str, archetype: str, lang: str = "ru") -> tuple[bool, float]:
    from difflib import SequenceMatcher

    h = hyp.strip()
    if not h or len(h) < 3:
        return False, 0.0
    cyr, heb, lat = _script_shares(h)
    # Prevent wrong-language ASR crashes (e.g. Croatian Latin for Russian, or Russian gibberish for Hebrew)
    if lang == "ru" and cyr < 0.55 and archetype != "address":
        return False, 0.0
    if lang == "mixed" and cyr < 0.35 and archetype != "address":
        return False, 0.0
    if lang == "en" and lat < 0.70:
        return False, 0.0
    if lang in ("he", "he_en") and heb < 0.50 and archetype != "address":
        return False, 0.0
    if lang == "he_ru" and (heb < 0.12 and cyr < 0.12):
        return False, 0.0

    # Prevent dropped clauses / runaway hallucinations
    tgt_words = [w for w in re.findall(r"\w+", target) if len(w) > 1]
    hyp_words = [w for w in re.findall(r"\w+", h) if len(w) > 1]
    if archetype not in ("address", "self_correction") and len(tgt_words) >= 5:
        wratio = len(hyp_words) / max(1, len(tgt_words))
        if wratio < 0.72 or wratio > 1.75:
            return False, 0.0

    fh = _phon_fold(h)
    fs = _phon_fold(spoken)
    ft = _phon_fold(target)
    sim = max(
        SequenceMatcher(None, fh, fs).ratio() if fs else 0.0,
        SequenceMatcher(None, fh, ft).ratio() if ft else 0.0,
    )
    if archetype == "address":
        min_sim = 0.60
    elif lang in ("he", "he_en", "he_ru"):
        min_sim = 0.68
    else:
        min_sim = 0.74
    if sim < min_sim:
        return False, round(sim, 3)

    # Check that long ordinary target words weren't silently dropped by TTS/ASR (prevents teaching hallucination!)
    if archetype not in ("address", "self_correction") and len(tgt_words) >= 4:
        fh_toks = fh.split()
        missed = 0
        checked = 0
        for w in tgt_words:
            if len(w) < 5 or any(ch.isdigit() for ch in w) or (w[0].isupper() and any("a" <= c <= "z" for c in w)):
                continue
            fw = _phon_fold(w)
            if len(fw) < 4:
                continue
            checked += 1
            best = max((SequenceMatcher(None, fw, ht).ratio() for ht in fh_toks), default=0.0)
            if best < 0.55 and fw[:4] not in fh:
                missed += 1
        if checked >= 3 and missed / checked > 0.22:
            return False, round(sim, 3)

    return True, round(sim, 3)


async def _edge_synth_one(text: str, voice: str, rate: str, pitch: str, tmp_dir: Path, tag: str) -> bytes | None:
    import edge_tts

    mp3 = tmp_dir / f"{tag}.mp3"
    wav = tmp_dir / f"{tag}.wav"
    for attempt in range(3):
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await comm.save(str(mp3))
            if mp3.exists() and mp3.stat().st_size > 200:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)],
                    check=True,
                )
                return wav.read_bytes()
        except Exception:
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


def _run_edge_batch(items: list[dict], tmp_dir: Path, max_conc: int = 10) -> dict[str, Path]:
    """Synthesize all Edge-TTS items concurrently to 16 kHz WAV files."""
    out_paths: dict[str, Path] = {}

    async def _go():
        sem = asyncio.Semaphore(max_conc)

        async def _one(it):
            async with sem:
                data = await _edge_synth_one(it["tts_input"], it["voice"], it["rate"], it["pitch"], tmp_dir, it["sid"])
                if data:
                    p = tmp_dir / f"{it['sid']}.wav"
                    p.write_bytes(data)
                    out_paths[it["sid"]] = p

        await asyncio.gather(*(_one(it) for it in items))

    asyncio.run(_go())
    return out_paths


# Worker globals for ProcessPoolExecutor (each process loads only the ASR/TTS models it needs)
_W_TTS: dict = {}
_W_ASR: dict = {}


def _worker_get_tts(engine: str):
    if engine not in _W_TTS:
        from ml.roundtrip.tts.registry import get_tts
        _W_TTS[engine] = get_tts(engine, num_threads=2)
    return _W_TTS[engine]


def _worker_get_asr(model_name: str):
    if model_name not in _W_ASR:
        from ml.roundtrip.asr.registry import get_driver
        drv = get_driver(model_name, num_threads=2)
        drv.load()
        _W_ASR[model_name] = drv
    return _W_ASR[model_name]


def _apply_speed(x, speed: float):
    if abs(speed - 1.0) < 0.01 or len(x) < 1600:
        return x
    import numpy as np
    from scipy.signal import resample

    new_len = max(1600, int(round(len(x) / speed)))
    return resample(x, new_len).astype(np.float32)


def _audio_worker_shard(shard: list[dict]) -> list[dict]:
    """Process a shard of items grouped by ASR model to keep RSS low and throughput high."""
    import numpy as np
    import soundfile as sf
    from ml.roundtrip.channel import apply_channel

    results = []
    for it in shard:
        sid = it["sid"]
        lang = it["lang"]
        eng = it["tts_engine"]
        voice = it["voice"]
        asr_model = it["asr_model"]
        preset = it["channel_preset"]
        speed = it["speed"]
        seed = it["seed"]

        try:
            if eng == "edge":
                wav_path = Path(it["edge_wav"])
                if not wav_path.exists():
                    results.append({"sid": sid, "ok": False, "error": "edge_missing"})
                    continue
                x, _ = sf.read(str(wav_path), dtype="float32")
            else:
                tts = _worker_get_tts(eng)
                tts_lang = "en" if lang == "en" else "ru"
                x = tts.synthesize(it["tts_input"], voice, tts_lang, seed=seed)
            if x is None or len(x) < 1600:
                results.append({"sid": sid, "ok": False, "error": "short_audio"})
                continue

            drv = _worker_get_asr(asr_model)
            asr_lang = "he" if lang in ("he", "he_en") else ("en" if lang == "en" else ("ru" if lang == "ru" else ""))

            # Try requested channel preset; if noisy audio wrecks intelligibility, retry cleanly once!
            chosen_preset = preset
            chosen_speed = speed
            hyp = ""
            sim = 0.0
            ok = False
            chosen_asr = asr_model
            for attempt_idx, (pr, sp) in enumerate([(preset, speed), ("clean", 1.0)]):
                if attempt_idx == 1 and pr == preset and sp == speed:
                    break
                sig = _apply_speed(x, sp)
                sig = apply_channel(sig, pr, seed=seed + attempt_idx)
                hyp = drv.transcribe(sig, asr_lang).strip()
                ok, sim = _asr_plausible(hyp, it["spoken"], it["target"], it["archetype"], lang)
                chosen_preset, chosen_speed = pr, sp
                if ok:
                    break

            # If a secondary ASR failed quality gate, give primary production ASR one clean try
            fallback_asr = ("zipformer2-66m-streaming-he@onnx-int8" if lang in ("he", "he_en")
                            else "parakeet-tdt-0.6b-v3@onnx-int8" if lang in ("ru", "en", "mixed")
                            else "")
            if not ok and fallback_asr and fallback_asr != asr_model:
                fb_drv = _worker_get_asr(fallback_asr)
                hyp2 = fb_drv.transcribe(x, asr_lang).strip()
                ok2, sim2 = _asr_plausible(hyp2, it["spoken"], it["target"], it["archetype"], lang)
                if ok2:
                    hyp, ok, sim = hyp2, ok2, sim2
                    chosen_asr, chosen_preset, chosen_speed = fallback_asr, "clean", 1.0

            results.append({
                "sid": sid,
                "ok": ok,
                "transcript": hyp,
                "sim": sim,
                "tts_engine": eng,
                "tts_voice": voice,
                "asr_model": chosen_asr,
                "channel": chosen_preset,
                "speed": chosen_speed,
                "filler_added": it["filler_added"],
                "duration_s": round(len(x) / 16000.0, 2),
                "error": "" if ok else f"low_sim:{sim}",
            })
        except Exception as exc:
            results.append({"sid": sid, "ok": False, "error": f"{type(exc).__name__}:{str(exc)[:140]}"})
    return results


def _load_teacher_examples(out_dir: Path) -> dict[str, dict]:
    from ml.data.synth.teacher import load_results

    done = load_results(out_dir / "teacher_shards", "gen")
    by_sid: dict[str, dict] = {}
    for row in done.values():
        res = row.get("result") or {}
        for ex in res.get("examples") or []:
            sid = ex.get("sid")
            if sid:
                by_sid[sid] = ex
    return by_sid


def run_audio(out_dir: Path, workers: int = 6, edge_conc: int = 12) -> None:
    scenarios = {
        s["sid"]: s
        for s in (json.loads(l) for l in (out_dir / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }
    teacher_by_sid = _load_teacher_examples(out_dir)
    audio_path = out_dir / "audio.jsonl"
    done_sids: set[str] = set()
    if audio_path.exists():
        for line in audio_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done_sids.add(r["sid"])

    todo = []
    for sid, ex in sorted(teacher_by_sid.items()):
        if sid in done_sids or sid not in scenarios:
            continue
        sc = scenarios[sid]
        spoken = (ex.get("spoken") or "").strip()
        target = (ex.get("target") or "").strip()
        tts_cyrl = (ex.get("tts_cyrl") or "").strip()
        if not spoken or not target:
            continue
        r = mkrng("audio_v3_tts", sid)
        eng, voice, rate, pitch = _choose_tts_plan(sc["lang"], bool(tts_cyrl), r)
        asr_model = _choose_asr_model(sc["lang"], r)
        preset, speed = _choose_channel(r)

        # Pick text fed to TTS
        base_tts = tts_cyrl if (eng == "piper" and sc["lang"] in ("ru", "mixed") and tts_cyrl) else spoken
        if sc["lang"] in ("he", "he_en", "he_ru"):
            base_tts = _clean_for_he_tts(base_tts)
        filler_added = False
        if sc["archetype"] not in ("no_op_and_clean", "address", "hebrew_hard"):
            base_tts, filler_added = _inject_spoken_filler(base_tts, sc["lang"], r)

        todo.append({
            "sid": sid,
            "lang": sc["lang"],
            "archetype": sc["archetype"],
            "spoken": spoken,
            "target": target,
            "tts_input": base_tts,
            "tts_engine": eng,
            "voice": voice,
            "rate": rate,
            "pitch": pitch,
            "asr_model": asr_model,
            "channel_preset": preset,
            "speed": speed,
            "filler_added": filler_added,
            "seed": sc["seed"],
        })

    print(f"[audio] teacher={len(teacher_by_sid)} done={len(done_sids)} todo={len(todo)}", flush=True)
    if not todo:
        return

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="murmur_v3_edge_") as tmp:
        tmp_dir = Path(tmp)
        # Step A: synthesize all Edge-TTS items concurrently first (network-bound, fast!)
        edge_items = [it for it in todo if it["tts_engine"] == "edge"]
        if edge_items:
            print(f"[audio] synthesizing {len(edge_items)} Edge-TTS clips (conc={edge_conc})...", flush=True)
            # Run in chunks of 200 to keep progress visible
            edge_paths: dict[str, Path] = {}
            for off in range(0, len(edge_items), 200):
                chunk = edge_items[off : off + 200]
                got = _run_edge_batch(chunk, tmp_dir, max_conc=edge_conc)
                edge_paths.update(got)
                print(f"[audio]   edge-tts {min(off + 200, len(edge_items))}/{len(edge_items)} (ok={len(edge_paths)}) [{time.time()-t0:.0f}s]", flush=True)
            for it in edge_items:
                p = edge_paths.get(it["sid"])
                it["edge_wav"] = str(p) if p else ""

        # Step B: group by ASR model into balanced shards so each worker only loads 1 ASR model per shard!
        by_asr: dict[str, list[dict]] = {}
        for it in todo:
            by_asr.setdefault(it["asr_model"], []).append(it)
        shards: list[list[dict]] = []
        for m, items in sorted(by_asr.items()):
            shard_sz = max(12, min(40, (len(items) + workers - 1) // workers))
            for off in range(0, len(items), shard_sz):
                shards.append(items[off : off + shard_sz])

        print(f"[audio] running {len(todo)} items across {len(shards)} ASR shards with {workers} workers...", flush=True)
        ok_n = fail_n = 0
        with open(audio_path, "a", encoding="utf-8") as fh:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_audio_worker_shard, sh) for sh in shards]
                for idx, fut in enumerate(as_completed(futs), 1):
                    res_list = fut.result()
                    for row in res_list:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if row.get("ok"):
                            ok_n += 1
                        else:
                            fail_n += 1
                    fh.flush()
                    if idx % 10 == 0 or idx == len(futs):
                        print(f"[audio]   shards {idx}/{len(futs)} ok={ok_n} fail={fail_n} [{time.time()-t0:.0f}s]", flush=True)
    print(f"[audio] DONE ok={ok_n} fail={fail_n} total_time={time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# 4. Chrome + 3-lang OCR Screen Assembly + Contract Export (run in data venv!)
# ---------------------------------------------------------------------------

def _render_screen_worker(items: list[dict], cache_dir_str: str) -> list[dict]:
    os.environ["MURMUR_CTX_CACHE"] = cache_dir_str
    from ml.data.synth import screen as SCR

    backend = SCR.Backend("real")
    out = []
    for it in items:
        sc = it["sc"]
        screen = it["screen"]
        sid = sc["sid"]
        if sc["screen_mode"] == "absent":
            r = mkrng("absent", sid)
            reason = r.choice(["secure_window", "denylisted_app", "screenshot_error", "ocr_empty"])
            ctx = {
                "v": 2,
                "app": dict(sc["app"]),
                "field": {"kind": sc["field_kind"], "hint": "", "before_cursor": ""},
            }
            out.append({"sid": sid, "ok": True, "ctx": ctx, "screen_meta": {"absent": reason}})
            continue

        # Build compatible sc dict for ml.data.synth.screen.make_spec
        sc_compat = {
            "sid": sid,
            "lang": sc["lang"],
            "category": "context_resolution" if sc["screen_must_show"] else "plain",
            "app": dict(sc["app"]),
            "field_kind": sc["field_kind"],
            "field_present": True,
            "screen_absent": None,
            "privacy": "full",
            "with_screen": sc["screen_mode"] != "empty",
            "render": SCR.render_params(sid, sc["app"]["category"], sc["lang"], True),
            "distractors": [],
            "spoken_terms": [{"text": t} for t in sc["said_terms"]],
            "people": [],
            "screen_terms": [],
            "screen_people": [],
            "injection": None,
        }
        if sc["screen_mode"] == "empty":
            screen = {"title": "", "messages": [], "before_cursor": "", "snippets": []}

        ok_ctx = None
        last_err = ""
        for variant in (0, 1):
            try:
                spec = SCR.make_spec(sc_compat, screen, variant)
                ctx = SCR.finalize(backend.build_ctx(spec), sc_compat)
                # Verify must_show and must_not_show against REAL OCR output!
                ok_vis = True
                for term in sc["screen_must_show"]:
                    if not SCR.on_screen(ctx, term):
                        ok_vis = False
                        last_err = f"ocr_missing:{term}"
                        break
                if ok_vis:
                    for term in sc["must_not_show"]:
                        if SCR.close_on_screen(ctx, term):
                            ok_vis = False
                            last_err = f"ocr_leaked:{term}"
                            break
                if ok_vis:
                    ok_ctx = (ctx, {"spec_template": spec.get("template"), "variant": variant, "backend": backend.version})
                    break
            except Exception as exc:
                last_err = f"render_err:{type(exc).__name__}:{str(exc)[:80]}"
        if ok_ctx:
            out.append({"sid": sid, "ok": True, "ctx": ok_ctx[0], "screen_meta": ok_ctx[1]})
        else:
            out.append({"sid": sid, "ok": False, "error": last_err})
    return out


_ARCH_TO_CAT = {
    "it_with_screen": "context_resolution",
    "it_custom_screen": "context_resolution",
    "it_no_screen_term": "plain",
    "near_miss": "near_miss",
    "self_correction": "self_correction",
    "hebrew_hard": "plain",
    "numbers_and_lists": "numbers",
    "address": "address",
    "dictated_question_or_prompt": "dictated_question",
    "everyday_diverse": "plain",
    "no_op_and_clean": "no_op",
}


def run_assemble(out_dir: Path, workers: int = 4) -> None:
    scenarios = {
        s["sid"]: s
        for s in (json.loads(l) for l in (out_dir / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }
    teacher_by_sid = _load_teacher_examples(out_dir)
    audio_by_sid: dict[str, dict] = {}
    for line in (out_dir / "audio.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                audio_by_sid[r["sid"]] = r

    valid_items = []
    for sid, aud in sorted(audio_by_sid.items()):
        if sid in scenarios and sid in teacher_by_sid:
            valid_items.append({"sc": scenarios[sid], "screen": teacher_by_sid[sid]["screen"]})

    print(f"[assemble] audio_ok={len(audio_by_sid)} rendering screens for {len(valid_items)} items (workers={workers})...", flush=True)
    cache_dir = out_dir / "ctx_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    chunks = [valid_items[i : i + 25] for i in range(0, len(valid_items), 25)]
    ctx_by_sid: dict[str, dict] = {}
    ocr_fails: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_render_screen_worker, ch, str(cache_dir)) for ch in chunks]
        for idx, fut in enumerate(as_completed(futs), 1):
            for res in fut.result():
                if res["ok"]:
                    ctx_by_sid[res["sid"]] = res
                else:
                    ocr_fails[res["error"].split(":")[0]] += 1
            if idx % 10 == 0 or idx == len(futs):
                print(f"[assemble]   chunks {idx}/{len(futs)} ctx_ok={len(ctx_by_sid)} ocr_fail={sum(ocr_fails.values())} [{time.time()-t0:.0f}s]", flush=True)

    # Assemble final train/dev/test JSONL records in exact Murmur contract format
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    seen_pairs: set[tuple[str, str]] = set()
    stats = {"by_split": Counter(), "by_lang": Counter(), "by_arch": Counter(), "by_asr": Counter(), "by_tts": Counter(), "ocr_fails": dict(ocr_fails)}

    for sid, ctx_res in sorted(ctx_by_sid.items()):
        sc = scenarios[sid]
        ex = teacher_by_sid[sid]
        aud = audio_by_sid[sid]
        transcript = aud["transcript"].strip()
        target = ex["target"].strip().replace("’", "'")
        if not transcript or not target:
            continue
        pair = (transcript.lower(), target.lower())
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cat = _ARCH_TO_CAT.get(sc["archetype"], "plain")
        if sc["archetype"] == "dictated_question_or_prompt" and sc["app"]["label"] in ("ChatGPT", "Claude", "Gemini"):
            cat = "ai_addressed"
        elif sc["archetype"] == "no_op_and_clean" and transcript != target:
            cat = "near_no_op"

        rec = {
            "id": stable_id("av3", sid),
            "lang": sc["lang"],
            "category": cat,
            "source": "tts",
            "split": sc["split"],
            "ctx": ctx_res["ctx"],
            "transcript": transcript,
            "target": target,
            "meta": {
                "scenario": sid,
                "group": f"av3-{sc['split']}-{sc['archetype']}-{sc['lang']}",
                "gen": "audio_v3.1",
                "archetype": sc["archetype"],
                "app_category": sc["app"]["category"],
                "app_label": sc["app"]["label"],
                "spoken": ex["spoken"],
                "tts_engine": aud["tts_engine"],
                "tts_voice": aud["tts_voice"],
                "asr_model": aud["asr_model"],
                "audio_channel": aud["channel"],
                "audio_speed": aud["speed"],
                "audio_sim": aud["sim"],
                "duration_s": aud["duration_s"],
                "said_terms": sc["said_terms"],
                "screen_must_show": sc["screen_must_show"],
                "screen": ctx_res["screen_meta"],
            },
        }
        by_split[sc["split"]].append(rec)
        stats["by_split"][sc["split"]] += 1
        stats["by_lang"][sc["lang"]] += 1
        stats["by_arch"][sc["archetype"]] += 1
        stats["by_asr"][aud["asr_model"]] += 1
        stats["by_tts"][aud["tts_engine"]] += 1

    for sp, rows in by_split.items():
        p = out_dir / f"{sp}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

    summary = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    (out_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[assemble] DONE: {json.dumps(summary, ensure_ascii=False)}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Girgur v3 real-audio synthetic dataset generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--dir", required=True)
    p_plan.add_argument("--count", type=int, default=4200)
    p_plan.add_argument("--seed", type=int, default=20261125)

    p_t = sub.add_parser("teacher")
    p_t.add_argument("--dir", required=True)
    p_t.add_argument("--batch-size", type=int, default=8)
    p_t.add_argument("--concurrency", type=int, default=6)
    p_t.add_argument("--max-requests", type=int, default=0)

    p_a = sub.add_parser("audio")
    p_a.add_argument("--dir", required=True)
    p_a.add_argument("--workers", type=int, default=6)
    p_a.add_argument("--edge-conc", type=int, default=12)

    p_s = sub.add_parser("assemble")
    p_s.add_argument("--dir", required=True)
    p_s.add_argument("--workers", type=int, default=4)

    args = ap.parse_args(argv)
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "plan":
        scs = make_plan(args.count, args.seed)
        (out_dir / "scenarios.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in scs) + "\n", encoding="utf-8"
        )
        print(f"[plan] wrote {len(scs)} scenarios to {out_dir / 'scenarios.jsonl'}", flush=True)
        print("  splits:", dict(Counter(s["split"] for s in scs)))
        print("  langs:", dict(Counter(s["lang"] for s in scs)))
        print("  archetypes:", dict(Counter(s["archetype"] for s in scs)))
        return 0
    if args.cmd == "teacher":
        run_teacher(out_dir, args.batch_size, args.concurrency, args.max_requests)
        return 0
    if args.cmd == "audio":
        run_audio(out_dir, args.workers, args.edge_conc)
        return 0
    if args.cmd == "assemble":
        run_assemble(out_dir, args.workers)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
