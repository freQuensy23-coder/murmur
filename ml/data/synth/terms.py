"""Hard IT/AI term pool for context-resolution cases.

Source: ml/roundtrip/probe/terms.jsonl (owned by ml/roundtrip, read-only here) -- the same terms the
round-trip probe speaks, so mined manglings exist for them -- plus lexicon.TECH as a fallback.

Every term has a split (train/dev/test) from a hash of its text: dev/test scenarios only use terms
never seen in training. `well_known` terms (GitHub, Kubernetes, Python...) may be restored by a model
without any context, so they never get a no-context counterpart.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

from ml.common import text as T
from ml.common.render_bridge import REPO
from ml.common.util import bucket_split
from ml.data.synth import lexicon as L

TERMS_JSONL = REPO / "ml" / "roundtrip" / "probe" / "terms.jsonl"
TERM_DEV, TERM_TEST = 0.12, 0.12
# Cyrillic jargon and shell commands are not "restore the Latin term from the screen" material.
EXCLUDED_CATEGORIES = {"dev_slang_ru", "ai_ml_jargon_ru", "cli_commands"}
WELL_KNOWN = {
    "GitHub", "GitLab", "Slack", "Telegram", "WhatsApp", "Zoom", "Python", "JavaScript", "TypeScript", "Java",
    "Android", "Linux", "Windows", "YouTube", "Instagram", "iPhone", "Excel", "PowerPoint", "ChatGPT", "Docker",
    "Kubernetes", "PostgreSQL", "Postgres", "Redis", "Kafka", "React", "Jira", "Figma", "Google Docs", "TikTok",
    "Netflix", "Spotify", "Samsung", "API", "GPU", "CPU", "SDK", "CUDA", "Google", "Apple", "Microsoft", "OpenAI",
    "NVIDIA", "MySQL", "MongoDB", "Node.js", "Kotlin", "Swift", "Rust", "Golang", "C++", "C#", "Django", "Flask",
    "Hugging Face", "PyTorch", "TensorFlow", "Jupyter", "Google Colab", "Notion", "Trello", "Confluence",
    "Jenkins", "Grafana", "Nginx", "Terraform", "Ansible", "Elasticsearch", "AWS", "Azure", "iOS", "macOS",
    "README", "JSON", "YAML", "HTML", "CSS", "SQL", "HTTP", "REST API", "Copilot", "GitHub Copilot", "Claude",
    # everyday acronyms and products a writer spells the usual way without any screen support (review 2026-11-18:
    # a counterpart that keeps 'о эр эм' for ORM teaches the wrong thing)
    "ORM", "CVE", "SaaS", "HTTPS", "URL", "PDF", "RAM", "ROM", "USB", "USB-C", "VPN", "UI", "UX", "CI", "CD", "CI/CD",
    "PR", "MR", "OOM", "LLM", "AI", "ML", "NLP", "SSH", "DNS", "IP", "TLS", "SSL", "JWT", "OAuth", "IDE", "CLI",
    "npm", "pip", "Git", "Wi-Fi", "Bluetooth", "Gmail", "Chrome", "Safari", "Firefox", "VS Code", "Xcode",
    "Android Studio", "Ubuntu", "MacBook", "iPad", "SSD", "HDD", "CRM", "ERP", "KPI", "OKR", "MVP", "QA", "DevOps",
    "SRE", "RAG", "GPT", "VRAM", "SDK", "APK", "XML", "CSV", "UTF-8", "HDMI", "NVMe", "Wi-Fi", "LTE", "5G", "SMS",
    "Google Drive", "Google Meet", "Teams", "Outlook", "Word", "Dropbox", "Bitbucket", "Stack Overflow", "Postman",
    "Swagger", "GraphQL", "gRPC", "WebSocket", "Redux", "Vue", "Angular", "Next.js", "Spring", "Laravel", "Unity",
    "agent", "ETA", "CDN", "Google Play",
    # 2025-2026 LLMs, coding agents, ML infra & modern dev tools that a smart editor spells right without a screen
    "Codex", "Claude Code", "Opus", "Sonnet", "Haiku", "Qwen", "Qwen3", "Qwen3.5", "GPT-4o", "GPT-5",
    "o3", "o3-mini", "o4-mini", "Gemini", "Gemini 2.5", "DeepSeek", "DeepSeek-R1", "Llama", "Llama 4",
    "Mistral", "Grok", "Perplexity", "Anthropic", "Midjourney", "Sora", "Whisper", "ElevenLabs",
    "Cursor", "Windsurf", "Devin", "MCP", "LoRA", "QLoRA", "GGUF", "ONNX", "vLLM", "llama.cpp", "Ollama",
    "LangChain", "LangGraph", "ComfyUI", "RunPod", "Modal", "WandB", "TensorRT", "Triton", "FlashAttention",
    "BFloat16", "FP8", "Q4_0", "INT8", "SFT", "RLHF", "DPO", "GRPO", "MoE",
    "FastAPI", "Pydantic", "Supabase", "Vercel", "Cloudflare", "Stripe", "Linear", "Sentry", "Datadog",
    "ClickHouse", "DuckDB", "Polars", "Tailscale", "WireGuard", "Tailwind", "Vite", "Bun", "uv", "ruff",
    "pytest", "Gradle", "Bazel", "Cargo", "Podman", "Helm", "ArgoCD", "Prometheus", "OpenTelemetry",
    "Playwright", "Cypress", "Prisma", "Drizzle", "tRPC", "Svelte", "Astro", "Flutter", "Jetpack Compose",
    "Ktor", "OkHttp", "Retrofit", "Protobuf", "WebRTC", "S3", "EC2", "EKS", "GCP", "BigQuery", "Vertex AI",
    "backend", "frontend", "fullstack", "deploy", "pipeline", "webhook", "microservice", "endpoint",
}


def _plain_english(t: str) -> bool:
    """Ordinary English words/phrases ('context window', 'throughput', 'distillation'): spelled right without a
    screen, so no no-context counterpart. Lowercase product names ('bitsandbytes', 'containerd') do not match."""
    return bool(re.fullmatch(r"[a-z]+( [a-z]+)+", t)) or bool(re.fullmatch(r"[a-z]+(tion|ing|put|ness|ment|ity|ance)", t))


@dataclass(frozen=True)
class Term:
    text: str
    category: str
    ru_spoken: tuple = ()
    en_spoken: tuple = ()
    well_known: bool = False

    @property
    def split(self) -> str:
        return bucket_split("term:" + self.text, dev=TERM_DEV, test=TERM_TEST)


def _usable(text: str) -> bool:
    # Latin-script tech/model names (digits allowed for Qwen3.5, GPT-5, RTX 5090, HTTP/2, etc.), short enough for context
    return bool(re.search(r"[A-Za-z]", text)) and len(text) <= 40 and not T.cyr_share(text)


@lru_cache(maxsize=1)
def all_terms() -> tuple[Term, ...]:
    out: dict[str, Term] = {}
    if TERMS_JSONL.exists():
        for line in TERMS_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") in EXCLUDED_CATEGORIES or not _usable(r["term"]):
                continue
            out[r["term"]] = Term(r["term"], r["category"], tuple(r.get("ru_spoken") or ()),
                                  tuple(r.get("en_spoken") or ()), r["term"] in WELL_KNOWN or _plain_english(r["term"]))
    for k, v in ({} if out else L.TECH).items():  # fallback only when the probe lexicon is absent
        if k not in out and _usable(k):
            out[k] = Term(k, "lexicon", tuple(v), (), k in WELL_KNOWN or k in L.KNOWN_TECH)
    return tuple(sorted(out.values(), key=lambda t: t.text))


@lru_cache(maxsize=None)
def distractor_pool_texts(split: str) -> frozenset:
    return frozenset(t.text for t in distractor_pool(split))


@lru_cache(maxsize=8)
def pool(split: str, hard_only: bool = False) -> tuple[Term, ...]:
    return tuple(t for t in all_terms() if t.split == split and (not hard_only or not t.well_known))


@lru_cache(maxsize=1)
def _index() -> dict:
    return {t.text: t for t in all_terms()}


def by_text(text: str) -> Term | None:
    return _index().get(text)


@lru_cache(maxsize=None)
def distractor_pool(split: str) -> tuple[Term, ...]:
    """Terms that may appear on screen as distractors in `split`. Train screens never show dev/test terms;
    held-out screens may show train terms as distractors (that makes them harder, not leakier: a held-out
    target term is still never seen in training)."""
    return pool("train") if split == "train" else pool(split) + pool("train")


NEAR_MISS_MIN = 0.6


def _score(a: str, b: str) -> float:
    """Look-alike score. Phonetic similarity only counts when both keys start alike and are >= 4 long (short
    keys make 'Qwen' ~ 'CVE'); a shared prefix of >= 3 letters counts ('ClickHouse' ~ 'ClickUp')."""
    al, bl = a.lower(), b.lower()
    pre = len(os.path.commonprefix([al, bl]))
    ka, kb = T.phonetic_key(a), T.phonetic_key(b)
    ph = T.phon_sim(a, b) if ka[:1] == kb[:1] and min(len(ka), len(kb)) >= 4 else 0.0
    lr = min(len(al), len(bl)) / max(1, max(len(al), len(bl)))
    return max(T.similarity(al, bl), ph * (0.7 + 0.3 * lr), (min(1.0, pre / 4) * 0.8) if pre >= 3 else 0.0)


def near_misses(term: Term, split: str, k: int = 2) -> list[Term]:
    """Terms that look or sound alike (ClickHouse ~ ClickUp, Kibana ~ Kubeflow), plus curated confusable pairs.
    Empty when nothing is close enough: a distractor that does not look like the term is not a near miss."""
    return [t for s, t in _ranked(term, split)[:k] if s >= NEAR_MISS_MIN]


@lru_cache(maxsize=None)
def _ranked(term: Term, split: str) -> tuple:
    cands = []
    curated = {b for a, b in L.CONFUSABLE if a == term.text} | {a for a, b in L.CONFUSABLE if b == term.text}
    allowed = distractor_pool_texts(split)
    for t in distractor_pool(split):
        if t.text == term.text or t.text.lower() in term.text.lower() or term.text.lower() in t.text.lower():
            continue
        s = _score(t.text, term.text) + (0.05 if t.category == term.category else 0.0) + (1.0 if t.text in curated else 0)
        cands.append((s, t.text, t))
    for c in curated - {t.text for t in all_terms()}:
        if split == "train" or c not in allowed:
            cands.append((1.0, c, Term(c, "curated")))
    cands.sort(key=lambda x: (-x[0], x[1]))
    return tuple((s, t) for s, _, t in cands[:5])
