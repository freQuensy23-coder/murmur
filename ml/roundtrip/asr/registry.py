"""Registry of the hub's CPU ASR models, resolved from ``docs/research/asr-catalog.json``.

Every model the hub ships with a CPU ONNX variant is registered here. Repos, file names and
pinned revisions come from the catalog. Two hub models whose catalog entries are NPU-only carry
their own pins to sherpa-onnx CPU exports of the same checkpoints (``_OFF_CATALOG``):
``whisper-medium@qnn-sm8750`` and ``nemotron-speech-streaming-en-0.6b@qnn-sm8750-560ms``.
``ml/roundtrip/mine/hub_map.json`` maps every hub catalog id to the profile id that stands in for it.

A model that cannot be resolved or loaded stays registered as an ``UnavailableDriver`` with the
reason, so it is reported instead of silently dropped.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence

from .base import AsrDriver, AsrSpec, UnavailableDriver
from .fetch import REPO_ROOT

CATALOG_PATH = REPO_ROOT / "docs" / "research" / "asr-catalog.json"

RU_EN: FrozenSet[str] = frozenset({"ru", "en"})
RU_EN_HE: FrozenSet[str] = frozenset({"ru", "en", "he"})
RU: FrozenSet[str] = frozenset({"ru"})
EN: FrozenSet[str] = frozenset({"en"})
HE: FrozenSet[str] = frozenset({"he"})

# name -> (kind, languages, role->file path inside the repo, extra recogniser kwargs)
_TRANSDUCER_INT8 = {"encoder": "encoder.int8.onnx", "decoder": "decoder.int8.onnx",
                    "joiner": "joiner.int8.onnx", "tokens": "tokens.txt"}
_MODELS: Dict[str, dict] = {
    "parakeet-tdt-0.6b-v3@onnx-int8": dict(kind="offline_transducer", langs=RU_EN, paths=_TRANSDUCER_INT8),
    "parakeet-ultra-0.6b@onnx-int8": dict(kind="offline_transducer", langs=RU_EN, paths=_TRANSDUCER_INT8),
    "nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms": dict(
        kind="online_transducer", langs=RU_EN, paths=_TRANSDUCER_INT8),
    "gigaam-v3-e2e-rnnt@onnx-int8": dict(
        kind="offline_transducer", langs=RU,
        paths={"encoder": "encoder.int8.onnx", "decoder": "decoder.onnx", "joiner": "joiner.onnx",
               "tokens": "tokens.txt"}),
    "gigaam-v3-e2e-ctc@onnx-int8": dict(
        kind="offline_nemo_ctc", langs=RU, paths={"model": "model.int8.onnx", "tokens": "tokens.txt"}),
    "t-one@onnx": dict(kind="online_t_one", langs=RU, paths={"model": "model.onnx", "tokens": "tokens.txt"}),
    "qwen3-asr-0.6b@onnx-int8": dict(
        kind="offline_qwen3", langs=RU_EN,
        paths={"conv_frontend": "conv_frontend.onnx", "encoder": "encoder.int8.onnx",
               "decoder": "decoder.int8.onnx", "tokenizer": "tokenizer"}),
    "whisper-large-v3-turbo@onnx-int8": dict(
        kind="offline_whisper", langs=RU_EN_HE,
        paths={"encoder": "turbo-encoder.int8.onnx", "decoder": "turbo-decoder.int8.onnx",
               "tokens": "turbo-tokens.txt"}),
    "distil-large-v3.5@onnx-int8": dict(
        kind="offline_whisper", langs=EN, default_lang="en",
        paths={"encoder": "distil-large-v3.5-encoder.int8.onnx",
               "decoder": "distil-large-v3.5-decoder.int8.onnx", "tokens": "distil-large-v3.5-tokens.txt"}),
    "whisper-small@onnx-int8": dict(
        kind="offline_whisper", langs=RU_EN_HE,
        paths={"encoder": "small-encoder.int8.onnx", "decoder": "small-decoder.int8.onnx",
               "tokens": "small-tokens.txt"}),
}

# Hub models the catalog lists only as SM8750 NPU binaries; the CPU path uses sherpa-onnx int8
# exports of the same checkpoints (revisions read from the HF API).
_OFF_CATALOG: Dict[str, dict] = {
    "zipformer2-66m-streaming-he@onnx-int8": dict(
        kind="online_transducer", langs=HE, default_lang="he",
        catalog_ref="zipformer2-66m-streaming-he@onnx-int8",
        repo="ivrit-ai/zipformer2-66m-streaming-he",
        revision="ee769a2a13b887f91572eec349458bfb0fe79e7a",
        paths={"encoder": "sherpa-onnx/encoder.int8.onnx", "decoder": "sherpa-onnx/decoder.int8.onnx",
               "joiner": "sherpa-onnx/joiner.int8.onnx", "tokens": "sherpa-onnx/tokens.txt"},
        kwargs={"model_type": "zipformer2"},
        licence="CC-BY-4.0 (ivrit-ai/zipformer2-66m-streaming-he)"),
    "nemotron-speech-streaming-en-0.6b@onnx-int8-560ms": dict(
        kind="online_transducer", langs=EN, default_lang="en",
        catalog_ref="nemotron-speech-streaming-en-0.6b@qnn-sm8750-560ms",
        repo="csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        revision="52056fdc070914a48dcd68b31b44d6a6f5b85902", paths=_TRANSDUCER_INT8,
        licence="NVIDIA Open Model License (nvidia/nemotron-speech-streaming-en-0.6b)"),
    "whisper-medium@onnx-int8": dict(
        kind="offline_whisper", langs=RU_EN, catalog_ref="whisper-medium@qnn-sm8750",
        repo="csukuangfj/sherpa-onnx-whisper-medium", revision="8c31d28503847560985df21f90e14f0c736e075e",
        paths={"encoder": "medium-encoder.int8.onnx", "decoder": "medium-decoder.int8.onnx",
               "tokens": "medium-tokens.txt"}, licence="Apache-2.0 (openai/whisper-medium)"),
}

ALL_MODELS: List[str] = list(_MODELS) + list(_OFF_CATALOG)

# Peak RSS (MB) of one worker process, 2 threads, measured on this server 2026-11-18 with
# ru_maxrss over 10 real round-trip clips (4-7 s) in one language, then rounded up ~15%.
# The runner's memory budget uses these to decide how many shards may run at once. Note that the
# RSS does not drop back after unload() (onnxruntime keeps its arena), hence worker recycling.
MEM_MB: Dict[str, int] = {
    "parakeet-tdt-0.6b-v3@onnx-int8": 1400,  # measured 1183
    "parakeet-ultra-0.6b@onnx-int8": 1400,  # 1195 (load + 3 clips)
    "nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms": 1150,  # 991
    "gigaam-v3-e2e-rnnt@onnx-int8": 500,  # 410
    "gigaam-v3-e2e-ctc@onnx-int8": 500,  # 423 (load + 3 clips)
    "t-one@onnx": 720,  # 615
    "qwen3-asr-0.6b@onnx-int8": 2000,  # 1715
    "whisper-large-v3-turbo@onnx-int8": 3400,  # 2985
    "distil-large-v3.5@onnx-int8": 2000,  # 1724
    "whisper-small@onnx-int8": 1500,  # 1302
    "whisper-medium@onnx-int8": 4000,  # 3499; 4544 with an en and a ru recogniser both loaded
    "nemotron-speech-streaming-en-0.6b@onnx-int8-560ms": 1150,  # same architecture as Nemotron 3.5
    "zipformer2-66m-streaming-he@onnx-int8": 300,
}

# Audio-seconds per wall-second of one 2-thread worker inside the 8-worker review run of 2026-11-19
# (server load average 23-44 from other jobs; an idle box is 3-4x faster). Only used to cut ASR
# shards to a similar expected wall time; a wrong value costs scheduling balance, not results.
XRT: Dict[str, float] = {
    "zipformer2-66m-streaming-he@onnx-int8": 8.0,
    "gigaam-v3-e2e-ctc@onnx-int8": 5.3,
    "gigaam-v3-e2e-rnnt@onnx-int8": 4.9,
    "parakeet-tdt-0.6b-v3@onnx-int8": 3.3,
    "parakeet-ultra-0.6b@onnx-int8": 3.2,
    "t-one@onnx": 3.0,
    "nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms": 0.92,
    "nemotron-speech-streaming-en-0.6b@onnx-int8-560ms": 0.92,  # not measured yet: same encoder size
    "qwen3-asr-0.6b@onnx-int8": 0.71,
    "whisper-small@onnx-int8": 0.66,
    "whisper-large-v3-turbo@onnx-int8": 0.49,
    "distil-large-v3.5@onnx-int8": 0.43,
    "whisper-medium@onnx-int8": 0.24,
}


@lru_cache(maxsize=1)
def load_catalog(path: Path = CATALOG_PATH) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {m["id"]: m for m in data["models"]}


class SpecError(RuntimeError):
    pass


def _files_for(name: str, cfg: dict, catalog: Dict[str, dict]) -> tuple[list, str, Optional[str], str]:
    """Return (hf files, family, catalog id, licence) for one registered model."""
    paths = cfg["paths"]
    if name in _OFF_CATALOG:
        files = [dict(repo=cfg["repo"], path=p, revision=cfg["revision"]) for p in paths.values()]
        family = catalog.get(cfg["catalog_ref"], {}).get("family", "openai-whisper")
        return files, family, None, cfg["licence"]
    entry = catalog.get(name)
    if entry is None:
        raise SpecError(f"{name} is not in {CATALOG_PATH.name}")
    by_path = {f["path"]: f for f in entry.get("files", []) if f.get("repo") and f.get("revision")}
    files = []
    for role, rel in paths.items():
        if role == "tokenizer":  # a directory: take every catalog file under it
            under = [f for p, f in by_path.items() if p.startswith(rel + "/")]
            if not under:
                raise SpecError(f"{name}: catalog has no files under {rel}/")
            files += [dict(repo=f["repo"], path=f["path"], revision=f["revision"]) for f in under]
            continue
        f = by_path.get(rel)
        if f is None:
            raise SpecError(f"{name}: catalog has no pinned file {rel!r}")
        files.append(dict(repo=f["repo"], path=f["path"], revision=f["revision"]))
    return files, entry.get("family", ""), name, entry.get("license", "")


def spec_for(name: str) -> AsrSpec:
    cfg = _MODELS.get(name) or _OFF_CATALOG.get(name)
    if cfg is None:
        raise KeyError(f"unknown ASR model {name!r}; known: {', '.join(ALL_MODELS)}")
    catalog = load_catalog()
    try:
        files, family, catalog_id, licence = _files_for(name, cfg, catalog)
        notes = ""
    except SpecError as exc:
        files, family, catalog_id, licence, notes = [], "", None, "", f"UNRESOLVED: {exc}"
    extra = dict(kind=cfg["kind"], paths=cfg["paths"], files=files, mem_mb=MEM_MB.get(name, 2000),
                 xrt=XRT.get(name, 1.0))
    if "kwargs" in cfg:
        extra["kwargs"] = cfg["kwargs"]
    if "default_lang" in cfg:
        extra["default_lang"] = cfg["default_lang"]
    if "catalog_ref" in cfg:
        extra["catalog_ref"] = cfg["catalog_ref"]
    return AsrSpec(name=name, family=family, languages=cfg["langs"], catalog_id=catalog_id,
                   notes=notes, licence=licence, extra=extra)


# Models known not to work on this machine, with the reason (filled from local verification).
KNOWN_UNAVAILABLE: Dict[str, str] = {}


def get_driver(name: str, num_threads: int = 2) -> AsrDriver:
    """Build the driver for ``name`` (nothing is downloaded or loaded until ``load()``)."""
    from .sherpa_drivers import SherpaDriver

    spec = spec_for(name)
    if spec.notes.startswith("UNRESOLVED"):
        return UnavailableDriver(spec, spec.notes, num_threads)
    if name in KNOWN_UNAVAILABLE:
        return UnavailableDriver(spec, KNOWN_UNAVAILABLE[name], num_threads)
    return SherpaDriver(spec, num_threads)


def all_specs() -> List[AsrSpec]:
    return [spec_for(n) for n in ALL_MODELS]


def resolve_models(selector: str | Sequence[str]) -> List[str]:
    """``"all"`` or a comma list / sequence of names (or unique substrings) -> registered names."""
    items: Iterable[str]
    if isinstance(selector, str):
        if selector.strip() == "all":
            return list(ALL_MODELS)
        items = [s for s in selector.split(",") if s.strip()]
    else:
        items = selector
    out: List[str] = []
    for raw in items:
        s = raw.strip()
        if s in ALL_MODELS:
            hit = [s]
        else:
            hit = [n for n in ALL_MODELS if s in n]
        if len(hit) != 1:
            raise KeyError(f"ASR selector {s!r} matches {hit or 'nothing'}; known: {', '.join(ALL_MODELS)}")
        if hit[0] not in out:
            out.append(hit[0])
    return out


DriverFactory = Callable[[str, int], AsrDriver]
