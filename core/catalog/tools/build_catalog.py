#!/usr/bin/env python3
"""Build core/catalog/src/main/resources/catalog/catalog.json (schema 1) from the verified research catalogs.

Inputs (read-only): docs/research/asr-catalog.json (verified by docs/research/verify/asr-ocr.md),
docs/research/llm-catalog.json (verified by docs/research/verify/llm-npu.md), the OCR model facts below
(docs/research/verify/asr-ocr.md section 2 + ml/ocr/README.md).

Every file is re-checked against the source API at build time, nothing is copied blindly:
  * Hugging Face: POST /api/models/<repo>/paths-info/<full revision> -> size + LFS sha256. Short revisions
    are resolved to the full commit sha. Small non-LFS files (tokens.txt, inference.yml) are downloaded
    once and hashed (cache: .cache/spikes/architect/catalog-files/).
  * GitHub release assets: GET /repos/<repo>/releases/tags/<tag> -> size (+ "digest" when GitHub
    provides it); the sha256 recorded by the asr-ocr spike (it re-hashed the tarballs) must match.
A mismatch aborts the build. Run:  python3 -I core/catalog/tools/build_catalog.py [--offline]
--offline skips the network checks and only re-shapes (use when the server has no network; say so).
"""
from __future__ import annotations

import hashlib
import json, re
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "core/catalog/src/main/resources/catalog/catalog.json"
CACHE = REPO / ".cache/spikes/architect/catalog-files"
ASR = json.loads((REPO / "docs/research/asr-catalog.json").read_text(encoding="utf-8"))
LLM = json.loads((REPO / "docs/research/llm-catalog.json").read_text(encoding="utf-8"))
OFFLINE = "--offline" in sys.argv
CATALOG_VERSION = "2026-11-22.1"

UA = {"User-Agent": "murmur-catalog-builder/1"}

# This server's IPv6 route to the HF CDN hangs in SYN-SENT; prefer IPv4 (curl falls back on its own).
import socket as _socket
_orig_gai = _socket.getaddrinfo


def _gai_v4(host, *a, **k):
    res = _orig_gai(host, *a, **k)
    v4 = [r for r in res if r[0] == _socket.AF_INET]
    return v4 or res


_socket.getaddrinfo = _gai_v4


def http_json(url: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


_rev_cache: dict[tuple[str, str], str] = {}


def full_revision(repo: str, rev: str) -> str:
    if len(rev) == 40 or OFFLINE:
        return rev
    key = (repo, rev)
    if key not in _rev_cache:
        _rev_cache[key] = http_json(f"https://huggingface.co/api/models/{repo}/revision/{rev}")["sha"]
        if all(c in "0123456789abcdef" for c in rev):
            assert _rev_cache[key].startswith(rev), (repo, rev, _rev_cache[key])
    return _rev_cache[key]


def hf_paths_info(repo: str, rev: str, paths: list[str]) -> dict[str, dict]:
    body = json.dumps({"paths": paths, "expand": False}).encode()
    rows = http_json(f"https://huggingface.co/api/models/{repo}/paths-info/{rev}", body,
                     {"Content-Type": "application/json"})
    return {r["path"]: r for r in rows}


def sha256_of_download(url: str) -> tuple[str, int]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / hashlib.sha1(url.encode()).hexdigest()
    if not f.exists():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            f.write_bytes(r.read())
    b = f.read_bytes()
    return hashlib.sha256(b).hexdigest(), len(b)


_gh_cache: dict[tuple[str, str], dict] = {}


def gh_asset(repo: str, tag: str, name: str) -> dict:
    key = (repo, tag)
    if key not in _gh_cache:
        _gh_cache[key] = {a["name"]: a for a in http_json(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")["assets"]}
    return _gh_cache[key][name]


def hf_file(repo: str, rev: str, path: str, *, bytes_: int | None = None, sha256: str | None = None,
            name: str | None = None) -> dict:
    rev = full_revision(repo, rev)
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    print("  check", repo, path, file=sys.stderr, flush=True)
    if not OFFLINE:
        info = hf_paths_info(repo, rev, [path])[path]
        size = info["size"]
        lfs = (info.get("lfs") or {}).get("oid")
        if lfs is None and size <= 8_000_000:
            h, n = sha256_of_download(url)   # small non-LFS file: hash it ourselves
            assert n == size, (url, n, size)
            lfs = h
        if lfs is None:
            # Xet-only storage exposes no sha256; trust the research catalog's value only if it has one.
            assert sha256, f"no sha256 obtainable for large file {url} ({size} B)"
            lfs = sha256
        if bytes_ is not None:
            assert bytes_ == size, f"size mismatch {url}: catalog {bytes_} api {size}"
        if sha256 is not None:
            assert sha256 == lfs, f"sha256 mismatch {url}: catalog {sha256} api {lfs}"
        bytes_, sha256 = size, lfs
    assert sha256, f"no sha256 for {url} (run without --offline)"
    return {"name": name or path.split("/")[-1], "url": url, "bytes": bytes_, "sha256": sha256,
            "source": "huggingface", "repo": repo, "revision": rev, "path": path}


def gh_file(repo: str, tag: str, path: str, bytes_: int, sha256: str, archive: str | None) -> dict:
    if not OFFLINE:
        a = gh_asset(repo, tag, path)
        assert a["size"] == bytes_, (path, a["size"], bytes_)
        dg = a.get("digest")
        if dg:
            assert dg == f"sha256:{sha256}", (path, dg, sha256)
    f = {"name": path, "url": f"https://github.com/{repo}/releases/download/{tag}/{path}", "bytes": bytes_,
         "sha256": sha256, "source": "github", "repo": repo, "revision": tag, "path": path}
    if archive:
        f["archive"] = archive
    return f


# WP-21: the ORT QNN plugin EP artifacts are our own host compiles (native/qnn-contexts/compile.sh, QAIRT 2.52 via
# onnxruntime-qnn 2.8.0, SM8750). Their bytes and sha256 come from the committed build manifest; QNN compiles are not
# bit-reproducible, so the pinned files are exactly the ones that build wrote (and the simulator suite checked).
# Hosting is the owner's HF repo (docs/OWNER_QUESTIONS.md, WP-21 answered 2026-11-20): the files are uploaded by
# tools/models/publish_npu_contexts.sh at the tagged revision named below, so the URLs answer and the network run
# checks size and sha256 of each. No shipped build offers these downloads without a -Pmurmur.qnn=true build:
# the ort-qnn runtimes are reported only by a -Pmurmur.qnn=true build whose QNN EP registered (RuntimeKeys.qnnEp,
# ADR-048/059).
QNN_CTX_MANIFEST = json.loads((REPO / "native/qnn-contexts/manifest.json").read_text(encoding="utf-8"))
NPU_CTX_REPO = "satoshiNakomoroReal/npu-contexts"
NPU_CTX_REVISION = "qairt-2.52.0-r1"
# NPU context files whose host did not answer (network run) or that were not checked (--offline): not verified.
NPU_CTX_PENDING: list[str] = []


def npu_ctx_files(artifact_id: str) -> tuple[list[dict], dict]:
    """FileSpecs of one compiled artifact (its zip, or its plain files) + the manifest entry."""
    m = QNN_CTX_MANIFEST["artifacts"][artifact_id]
    plain = m.get("archive") is None
    rows = m["files"] if plain else [m["archive"]]
    out = []
    for r in rows:
        path = f"{artifact_id.replace('@', '--')}/{r['name']}"
        url = f"https://huggingface.co/{NPU_CTX_REPO}/resolve/{NPU_CTX_REVISION}/{path}"
        info = None
        if not OFFLINE:
            try:
                info = hf_paths_info(NPU_CTX_REPO, NPU_CTX_REVISION, [path]).get(path)
            except urllib.error.HTTPError as e:
                print(f"  pending host {NPU_CTX_REPO}@{NPU_CTX_REVISION}: HTTP {e.code} for {path}", file=sys.stderr, flush=True)
            if info is not None:
                lfs = (info.get("lfs") or {}).get("oid")
                assert info["size"] == r["bytes"] and lfs in (None, r["sha256"]), (path, info, r)
        if info is None:
            NPU_CTX_PENDING.append(path)
        f = {"name": r["name"], "url": url, "bytes": r["bytes"], "sha256": r["sha256"], "source": "huggingface",
             "repo": NPU_CTX_REPO, "revision": NPU_CTX_REVISION, "path": path}
        if not plain:
            f["archive"] = "zip"
        out.append(f)
    return out, m


def npu_ctx_artifact(artifact_id: str, runtime: str, *, window: float | None = None, notes: str) -> dict:
    files, m = npu_ctx_files(artifact_id)
    v = QNN_CTX_MANIFEST["versions"]
    mb = -(-sum(f["bytes"] for f in files) * 3 // (2 * 1024 * 1024))
    a = {"id": artifact_id, "runtime": runtime, "format": "onnx-epcontext", "quant": "fp16", "units": ["npu"],
         "socs": [v["soc"]], "files": files, "layout": m["layout"], "memoryMb": {"npu": mb},
         "notes": f"{notes} Host AOT compile: onnxruntime-qnn {v['onnxruntime_qnn']} (QAIRT {v['qairt']}), HTP v{v['htp_arch']}, "
                  "100 % on the HTP (CPU fallback disabled). Off by default: adopted only after DEVICE_CHECKLIST DC4 (ADR-059). "
                  "memoryMb estimated (download bytes x 1.5). Hosting pending (owner)."}
    if window:
        a["maxWindowSec"] = window
    return a


def from_asr_files(src: dict) -> list[dict]:
    out = []
    for f in src["files"]:
        if f["source"] == "huggingface":
            out.append(hf_file(f["repo"], f["revision"], f["path"], bytes_=f["bytes"], sha256=f.get("sha256")))
        elif f["source"] == "github-release":
            out.append(gh_file(f["repo"], f["release"], f["path"], f["bytes"], f["sha256"], f.get("archive")))
        else:
            raise ValueError(f"unsupported source {f['source']} in {src['id']}")
    return out


ASR_BY_ID = {m["id"]: m for m in ASR["models"] + ASR["auxiliary"]}
LLM_BY_ID = {m["id"]: m for m in LLM["models"]}


def langs(src: dict) -> dict:
    l = src.get("languages") or {}
    codes = l.get("all") or l.get("transcriptionReady") or []
    mode = "auto" if l.get("auto") else "fixed"
    return {"codes": codes, "count": l.get("count", len(codes)), "mode": mode, **({"note": l["notes"]} if l.get("notes") else {})}


def lic(src: dict) -> dict:
    v = src.get("license")
    if isinstance(v, dict):
        return {"id": v["id"], **({"url": v["url"]} if v.get("url") else {}), **({"note": v["commercial"]} if v.get("commercial") else {})}
    return {"id": v or "unknown", **({"note": src["licenseNotes"]} if src.get("licenseNotes") else {})}


def asr_artifact(src_id: str, runtime: str, units: list[str], *, layout: dict, window: float | None = None,
                 memory: dict | None = None, notes: str | None = None, quant: str | None = None) -> dict:
    s = ASR_BY_ID[src_id]
    a = {"id": src_id, "runtime": runtime, "format": s.get("format", "onnx"), "units": units,
         "files": from_asr_files(s), "layout": layout}
    if quant:
        a["quant"] = quant
    if s.get("socs"):
        a["socs"] = s["socs"]
    if window:
        a["maxWindowSec"] = window
    if not memory:
        # MemoryEstimate fallback (ADR-045 review): file bytes x 1.5, so that every bundled artifact carries a value.
        mb = -(-sum(f["bytes"] for f in a["files"]) * 3 // (2 * 1024 * 1024))
        memory = {u: mb for u in units}
        notes = ((notes + " ") if notes else "") + "memoryMb estimated (download bytes x 1.5)."
    a["memoryMb"] = memory
    if notes:
        a["notes"] = notes
    return a


TRANSDUCER_CPU = {"encoder": "encoder.int8.onnx", "decoder": "decoder.int8.onnx", "joiner": "joiner.int8.onnx", "tokens": "tokens.txt"}
# Inner paths of the sherpa SM8750 tarballs: VERIFIED by streaming `curl | bzip2 -dc | tar -tv` (architect, 2026-11-18;
# listings in .cache/spikes/architect/tarlist-*.txt) for Parakeet v3 20 s, Nemotron 3.5 560 ms and Whisper small.
# Whisper medium and Nemotron EN are assumed to follow the same pattern (UNVERIFIED; WP-06 checks at extraction).
QNN_PARAKEET = {"dir": "sherpa-onnx-qnn-SM8750-binary-parakeet-tdt-0.6b-v3-20s-transducer", "encoder": "encoder.bin",
                "decoder": "decoder.bin", "joiner": "joiner.bin", "tokens": "tokens.txt"}


def asr_models() -> list[dict]:
    M = []
    ev = "docs/research/verify/asr-ocr.md"

    def model(id_, family, name, publisher, base_src, traits, artifacts, tier, desc, follow=None, notes=None):
        s = ASR_BY_ID[base_src]
        m = {"id": id_, "kind": "asr", "displayName": name, "publisher": publisher, "family": family,
             "baseRepo": s.get("baseModel"), "description": desc, "params": s.get("params"),
             "paramsLabel": s.get("paramsLabel"), "languages": langs(s), "license": lic(s), "tier": tier,
             "asr": traits, "artifacts": artifacts, "evidence": ev}
        if follow:
            m["followUps"] = follow
        if notes:
            m["notes"] = notes
        M.append(m)

    std = {"streaming": False, "punctuation": True, "casing": True}
    model("parakeet-tdt-0.6b-v3", "parakeet", "Parakeet TDT 0.6B v3", "NVIDIA", "parakeet-tdt-0.6b-v3@onnx-int8", std, [
        asr_artifact("parakeet-tdt-0.6b-v3@qnn-sm8750-20s", "sherpa/offline-transducer", ["npu"], layout=QNN_PARAKEET,
                     window=20.0, memory={"npu": 900}, quant="w8a16 (uq16 I/O)",
                     notes="sherpa-onnx SM8750 context binaries, QAIRT 2.40; loads with qnn-runtime 2.52 (simulator). NPU speed/accuracy UNVERIFIED-ON-DEVICE."),
        asr_artifact("parakeet-tdt-0.6b-v3@onnx-int8", "sherpa/offline-transducer", ["cpu"], layout=TRANSDUCER_CPU,
                     memory={"cpu": 1302}, quant="int8"),
    ], "recommended", "Russian and English with punctuation; follows language switches between phrases.",
        notes="Default ru+en: host FLEURS WER ru 7.4 / en 7.6 %, mixed ru/en with VAD segments 4.5 %.")
    model("parakeet-ultra-0.6b", "parakeet", "Parakeet Ultra 0.6B", "NVIDIA", "parakeet-ultra-0.6b@onnx-int8", std, [
        asr_artifact("parakeet-ultra-0.6b@onnx-int8", "sherpa/offline-transducer", ["cpu"], layout=TRANSDUCER_CPU, quant="int8"),
    ], "optional", "Parakeet variant, CPU build.",
        follow=[{"id": "npu-port", "note": "No SM8750 context binary published; compile with QAIRT 2.52 if it proves better than v3."}])
    model("gigaam-v3-e2e-rnnt", "gigaam", "GigaAM v3 RNN-T", "SberDevices", "gigaam-v3-e2e-rnnt@onnx-int8", std, [
        asr_artifact("gigaam-v3-e2e-rnnt@onnx-int8", "sherpa/offline-transducer", ["cpu"],
                     layout={"encoder": "encoder.int8.onnx", "decoder": "decoder.onnx", "joiner": "joiner.onnx", "tokens": "tokens.txt"},
                     window=20.0, memory={"cpu": 414}, quant="int8"),
    ], "recommended", "Best Russian accuracy; Russian only.",
        follow=[{"id": "npu-port", "note": "CTC variant compiles 100 % onto the HTP (ORT QNN EP, 20 s fixed shape); needs mel front end + CTC decode in the app. Decide after phone CPU numbers."}],
        notes="Host FLEURS WER ru 4.2 %; English is not supported (71 % WER).")
    model("gigaam-v3-e2e-ctc", "gigaam", "GigaAM v3 CTC", "SberDevices", "gigaam-v3-e2e-ctc@onnx-int8", std, [
        asr_artifact("gigaam-v3-e2e-ctc@onnx-int8", "sherpa/offline-ctc", ["cpu"],
                     layout={"model": "model.int8.onnx", "tokens": "tokens.txt"}, window=20.0, memory={"cpu": 401}, quant="int8"),
        npu_ctx_artifact("gigaam-v3-e2e-ctc@qnn-sm8750-20s", "ort-qnn/gigaam-ctc", window=20.0,
                         notes="fp32 istupakov/gigaam-v3-onnx v3_e2e_ctc.onnx fixed to [1,64,2000] (20 s); 64-bin mel front end "
                               "and greedy CTC in the app (WP-21)."),
    ], "optional", "Russian only, CTC decoder.",
        follow=[{"id": "npu-port", "note": "fp16 [1,64,2000] compiles 100 % onto the HTP (asr-ocr.md 5.4)."}])
    model("t-one", "t-one", "T-one", "T-Bank", "t-one@onnx", {"streaming": True, "punctuation": False, "casing": False, "sampleRate": 8000}, [
        asr_artifact("t-one@onnx", "sherpa/online-t-one", ["cpu"], layout={"model": "model.onnx", "tokens": "tokens.txt"}, memory={"cpu": 558}),
    ], "optional", "Russian streaming, lowercase without punctuation.",
        follow=[{"id": "npu-port", "note": "Streaming graph does not compile strictly for HTP (19 partitions + CPU BatchNorm); CPU only for now."}])
    model("nemotron-3.5-asr-streaming-0.6b", "nemotron", "Nemotron 3.5 ASR Streaming", "NVIDIA",
          "nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms", {**std, "streaming": True, "previewOnly": True}, [
        asr_artifact("nemotron-3.5-asr-streaming-0.6b@qnn-sm8750-560ms", "sherpa/online-transducer", ["npu"],
                     layout={"dir": "sherpa-onnx-qnn-SM8750-binary-nemotron-3.5-asr-streaming-0.6b-560ms", "encoder": "encoder.bin",
                             "decoder": "decoder.bin", "joiner": "joiner.bin", "tokens": "tokens.txt"}, memory={"npu": 900}),
        asr_artifact("nemotron-3.5-asr-streaming-0.6b@onnx-int8-560ms", "sherpa/online-transducer", ["cpu"], layout=TRANSDUCER_CPU,
                     memory={"cpu": 977}, quant="int8"),
    ], "optional", "Live preview while you speak; the final text comes from the selected batch model.",
        notes="Host WER ru 13.9 / en 12.8 %; one stream does not follow ru/en switches (33 % on the mixed clip).")
    model("nemotron-speech-streaming-en-0.6b", "nemotron", "Nemotron Speech Streaming EN", "NVIDIA",
          "nemotron-speech-streaming-en-0.6b@qnn-sm8750-560ms", {**std, "streaming": True, "previewOnly": True}, [
        asr_artifact("nemotron-speech-streaming-en-0.6b@qnn-sm8750-560ms", "sherpa/online-transducer", ["npu"],
                     layout={"dir": "sherpa-onnx-qnn-SM8750-binary-nemotron-speech-streaming-en-0.6b-560ms", "encoder": "encoder.bin",
                             "decoder": "decoder.bin", "joiner": "joiner.bin", "tokens": "tokens.txt"}),
    ], "optional", "English live preview.")
    wl = {"encoder": "small-encoder.int8.onnx", "decoder": "small-decoder.int8.onnx", "tokens": "small-tokens.txt"}
    model("whisper-small", "whisper", "Whisper Small", "OpenAI", "whisper-small@onnx-int8", std, [
        asr_artifact("whisper-small@qnn-sm8750", "sherpa/offline-whisper", ["npu"],
                     layout={"dir": "sherpa-onnx-qnn-SM8750-binary-whisper-small", "encoder": "encoder.bin", "decoder": "decoder.bin",
                             "tokens": "tokens.txt"}, window=30.0),
        asr_artifact("whisper-small@onnx-int8", "sherpa/offline-whisper", ["cpu"], layout=wl, window=30.0, memory={"cpu": 1202}, quant="int8"),
    ], "optional", "99 languages; 30 s per call.")
    model("whisper-medium", "whisper", "Whisper Medium", "OpenAI", "whisper-medium@qnn-sm8750", std, [
        asr_artifact("whisper-medium@qnn-sm8750", "sherpa/offline-whisper", ["npu"],
                     layout={"dir": "sherpa-onnx-qnn-SM8750-binary-whisper-medium", "encoder": "encoder.bin", "decoder": "decoder.bin",
                             "tokens": "tokens.txt"}, window=30.0),
    ], "optional", "99 languages, NPU only.")
    model("whisper-large-v3-turbo", "whisper", "Whisper Large v3 Turbo", "OpenAI", "whisper-large-v3-turbo@onnx-int8", std, [
        asr_artifact("whisper-large-v3-turbo@onnx-int8", "sherpa/offline-whisper", ["cpu"],
                     layout={"encoder": "turbo-encoder.int8.onnx", "decoder": "turbo-decoder.int8.onnx", "tokens": "turbo-tokens.txt"},
                     window=30.0, memory={"cpu": 2067}, quant="int8", notes="Host RTF 1.2: slower than real time on CPU."),
    ], "experimental", "Most accurate Whisper; too slow on CPU.",
        follow=[{"id": "npu-port", "note": "Qualcomm AI Hub w8a16 SM8750 context binary exists (QAIRT 2.50, s3 host); needs a custom runner (I/O differs from sherpa's)."}])
    model("distil-large-v3.5", "whisper", "Distil-Whisper Large v3.5", "Hugging Face", "distil-large-v3.5@onnx-int8", std, [
        asr_artifact("distil-large-v3.5@onnx-int8", "sherpa/offline-whisper", ["cpu"],
                     layout={"encoder": "distil-large-v3.5-encoder.int8.onnx", "decoder": "distil-large-v3.5-decoder.int8.onnx",
                             "tokens": "distil-large-v3.5-tokens.txt"}, window=30.0, quant="int8"),
    ], "experimental", "English-focused distilled Whisper.",
        follow=[{"id": "npu-port", "note": "No SM8750 binary published."}])
    qa = ASR_BY_ID["qwen3-asr-0.6b@onnx-int8"]
    model("qwen3-asr-0.6b", "qwen3-asr", "Qwen3-ASR 0.6B", "Qwen", "qwen3-asr-0.6b@onnx-int8", std, [
        asr_artifact("qwen3-asr-0.6b@onnx-int8", "sherpa/offline-qwen3-asr", ["cpu"],
                     layout={"frontend": "conv_frontend.onnx", "encoder": "encoder.int8.onnx", "decoder": "decoder.int8.onnx",
                             "merges": "merges.txt", "tokenizerConfig": "tokenizer_config.json", "vocab": "vocab.json"},
                     memory={"cpu": 2390}, quant="int8"),
        asr_artifact("qwen3-asr-0.6b@litert-npu-sm8750", "litert/npu", ["npu"], layout={},
                     notes="LiteRT NPU artifact exists (HEAD 200); LiteRT is not shipped in this version."),
    ], "experimental", "30 languages; weakest Russian in the host eval (19.9 % WER).")
    model("qwen3-asr-1.7b", "qwen3-asr", "Qwen3-ASR 1.7B", "Qwen", "qwen3-asr-1.7b@litert-gpu-15s", std, [
        asr_artifact("qwen3-asr-1.7b@litert-gpu-15s", "litert-lm/gpu", ["gpu"], layout={}, window=15.0,
                     notes="LiteRT-LM GPU build; runtime not shipped in this version."),
    ], "experimental", "Larger Qwen3-ASR, GPU build only.",
        follow=[{"id": "npu-port", "note": "No NPU artifact."}])
    # qwen3-asr CPU file names collide across tokenizer/ dirs; keep the basename (unique here).
    return M


# The OCR models are validated against ml/ocr's golden suite of this pipeline version (read, never hard-coded).
OCR_PIPELINE_VERSION = re.search(r'^PIPELINE_VERSION = "([^"]+)"', (REPO / "ml/ocr/config.py").read_text(encoding="utf-8"), re.M).group(1)


def aux_models() -> list[dict]:
    s = ASR_BY_ID["silero-vad@onnx-v5"]
    vad = {"id": "silero-vad-v5", "kind": "vad", "displayName": "Silero VAD v5", "publisher": "Silero", "family": "silero-vad",
           "description": "Finds pauses so speech can be transcribed while you talk.", "license": {"id": "MIT"},
           "tier": "recommended", "artifacts": [{
               "id": "silero-vad-v5@onnx", "runtime": "ort/silero-vad-v5", "format": "onnx", "units": ["cpu"],
               "files": from_asr_files(s), "layout": {"model": "silero_vad_v5.onnx"}, "memoryMb": {"cpu": 30}}],
           "evidence": "docs/research/verify/asr-ocr.md"}
    det = {"id": "pp-ocrv6-tiny-det", "kind": "ocr-det", "displayName": "PP-OCRv6 tiny detector", "publisher": "PaddlePaddle",
           "family": "pp-ocr", "baseRepo": "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
           "description": "Finds text lines in the screenshot.", "license": {"id": "Apache-2.0"}, "tier": "recommended",
           "ocr": {"stage": "det", "pipelineVersion": OCR_PIPELINE_VERSION},
           "artifacts": [{"id": "pp-ocrv6-tiny-det@onnx", "runtime": "ort/ppocr-det", "format": "onnx", "units": ["cpu"],
                          "files": [hf_file("PaddlePaddle/PP-OCRv6_tiny_det_onnx", "2ba1506", "inference.onnx",
                                            sha256="193bab7a" if OFFLINE else None, name="det.onnx")],
                          "layout": {"model": "det.onnx"}, "memoryMb": {"cpu": 60}},
                         npu_ctx_artifact("pp-ocrv6-tiny-det@qnn-sm8750", "ort-qnn/ppocr-det",
                                          notes="One context per detector input 960x416, 416x960, 960x864, 864x960, 960x960; any other window shape runs the unchanged det.onnx shipped in the same zip on the CPU (padding into a larger context changes the output: the detector pools globally).")],
           "followUps": [{"id": "npu-port", "note": "fp16 fixed-shape graphs compile 100 % onto the HTP (asr-ocr.md 7.5); decide after phone CPU vs QNN times."}],
           "evidence": "docs/research/verify/asr-ocr.md"}
    rec = {"id": "pp-ocrv5-cyrillic-mobile-rec", "kind": "ocr-rec", "displayName": "PP-OCRv5 Cyrillic recogniser",
           "publisher": "PaddlePaddle", "family": "pp-ocr", "baseRepo": "PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx",
           "description": "Reads Cyrillic and Latin text lines.", "license": {"id": "Apache-2.0"}, "tier": "recommended",
           "languages": {"codes": ["ru", "en", "uk", "be", "bg", "sr"], "count": 6, "mode": "auto"},
           "ocr": {"stage": "rec", "pipelineVersion": OCR_PIPELINE_VERSION},
           "artifacts": [{"id": "pp-ocrv5-cyrillic-mobile-rec@onnx", "runtime": "ort/ppocr-rec", "format": "onnx", "units": ["cpu"],
                          "files": [hf_file("PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx", "2cef881", "inference.onnx", name="rec.onnx"),
                                    hf_file("PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx", "2cef881", "inference.yml", name="rec.yml")],
                          "layout": {"model": "rec.onnx", "dict": "rec.yml"}, "memoryMb": {"cpu": 80}},
                         npu_ctx_artifact("pp-ocrv5-cyrillic-mobile-rec@qnn-sm8750", "ort-qnn/ppocr-rec",
                                          notes="One context per ml/ocr bucket width 320..3200 at height 48, batch 1 (padding changes the output, so every bucket has its own).")],
           "followUps": [{"id": "npu-port", "note": "Width buckets 320/640/1280 compile onto the HTP in fp16 (asr-ocr.md 7.5)."}],
           "evidence": "docs/research/verify/asr-ocr.md"}
    return [vad, det, rec]


LLM_TEMPLATE = {"lfm3-chatml": "lfm3", "lfm2-chatml": "lfm2.5", "lfm2-chatml-always-think": "lfm2.5",
                "qwen3.5-chatml": "qwen3.5", "qwen3-chatml": "qwen3"}
# LLM spike D4 (docs/research/verify/llm-npu.md): default unit before any benchmark.
PREFERRED = {"lfm3": "npu", "lfm2.5": "npu", "qwen3.5": "cpu", "qwen3": "npu"}
# Host no-mmap peak RSS where measured (llm-npu.md 5.4), else the research estimate.
CPU_MB = {"lfm3-1.7b": 1250, "qwen3.5-0.8b": 710}
# x86 emulator peak RSS at n_ctx 9,216 with the prefix built (WP-10 LlmServiceTest, 1,474 / 873 MB), rounded up:
# a lower bound on the CPU memoryMb until DC2 measures the phone (verification F4: a gate that trusted 1,310 / 770
# could start a load that lmkd then kills). Kept as the floor after the stock nCtx shrank to 7,168 (ADR-085): the
# smaller KV only makes the real peak lower, and DC2 replaces these numbers with the phone's.
CPU_PEAK_MB = {"lfm3-1.7b": 1480, "qwen3.5-0.8b": 880}
# ADR-045: tokens of render.cache_prefix(profile, template) with the host llama-tokenize (b12503, --no-bos) on the
# real GGUFs. default.v4 re-measured 2026-11-19 by native/llama/tools/tokenizer_check.py (WP-10); default.v1 measured
# 2026-11-20 the same way when it became the stock profile (ADR-085, docs/ml/PROMPT_EVAL.md section R). default.v4 and
# default.v4s stay here as measured, selectable profiles. LFM2.5-2.6B has its own vocabulary (different tokenizer
# fingerprint from 1.2B/350M), hence the per-model override.
# tokenizer_check.py fails when a prompt, profile or template change moves any catalog number.
STOCK_PROFILE = "default.v1"  # = GgufTemplates.STOCK_PROFILE (ADR-085)
PREFIX_TOKENS = {("default.v1", "qwen3.5"): 1047, ("default.v1", "lfm3"): 1072, ("default.v1", "lfm2.5"): 1101,
                 ("default.v1", "qwen3"): 1062,
                 ("default.v4", "qwen3.5"): 3116, ("default.v4", "lfm3"): 3226, ("default.v4", "lfm2.5"): 3256,
                 ("default.v4", "qwen3"): 3146, ("default.v4s", "qwen3.5"): 2001, ("default.v4s", "lfm3"): 2060,
                 ("default.v4s", "lfm2.5"): 2079, ("default.v4s", "qwen3"): 2024,
                 ("finetuned.v1", "qwen3.5"): 97, ("finetuned.v1", "lfm3"): 102,
                 ("finetuned.v1", "lfm2.5"): 102, ("finetuned.v1", "qwen3"): 97}
PREFIX_TOKENS_MODEL = {("default.v1", "lfm2.5-2.6b"): 1072, ("default.v4", "lfm2.5-2.6b"): 3196,
                       ("default.v4s", "lfm2.5-2.6b"): 2044}
CONTEXT_RESERVE_TOKENS = 5200  # = LlmTraits.CONTEXT_RESERVE_TOKENS (5-minute dictation, ADR-045)
KV_DEFAULT = 12288


def n_ctx_for(prefix_tokens: int) -> int:
    need = prefix_tokens + CONTEXT_RESERVE_TOKENS
    return (need + 1023) // 1024 * 1024


def kv_mb(n_ctx: int, kv_bytes: int | None) -> int:
    return -(-n_ctx * (kv_bytes or KV_DEFAULT) // (1024 * 1024))


def finetune_2b(src: dict) -> dict:
    """review 2026-11-19 (ADR-046): ML trains Qwen3.5 0.8B/2B and LFM3 (ml/eval/contract.py TEMPLATES), not Qwen3."""
    d = dict(src)
    d.update({"id": "murmur-cleanup-qwen3.5-2b", "display_name": "Murmur Cleanup (Qwen3.5 2B)", "base_repo": "Qwen/Qwen3.5-2B",
              "params": 2274069824, "notes": "Phase B of docs/ml/TRAINING_RUNBOOK.md; Q4_0 GGUF."})
    return d


def llm_models() -> list[dict]:
    out = []
    src_models = []
    for m in LLM["models"]:
        if m["id"] == "murmur-cleanup-qwen3-0.6b":
            continue  # cannot be produced: the training contract has no qwen3 template (ADR-046)
        src_models.append(m)
        if m["id"] == "murmur-cleanup-qwen3.5-0.8b":
            src_models.append(finetune_2b(m))
    for m in src_models:
        if m["id"] == "none":
            continue  # "None" is a UI choice, not a catalog model
        fam = m["family"]
        family = "murmur-cleanup" if fam == "finetune" else fam
        tmpl = LLM_TEMPLATE.get(m.get("template_id", ""), "qwen3.5")
        planned = m.get("status") == "planned"
        e = {"id": m["id"], "kind": "llm", "displayName": m["display_name"],
             "publisher": "Murmur" if planned else ("Liquid AI" if fam.startswith("lfm") else "Qwen"),
             "family": family, "baseRepo": m.get("base_repo"), "params": m.get("params"),
             "paramsLabel": m.get("params_display"),
             "languages": {"codes": [c for c in m.get("languages", []) if len(c) <= 3], "mode": "auto",
                           "count": 201 if fam.startswith("qwen3.5") else len([c for c in m.get("languages", []) if len(c) <= 3]),
                           **({"note": m["russian"]} if m.get("russian") else {})},
             "license": lic(m),
             "tier": {"default": "recommended", "option": "optional", "experimental": "experimental"}.get(m.get("role"), "planned"),
             "status": "in_training" if planned else "available",
             "llm": {"template": tmpl, "promptProfile": (profile := "finetuned.v1" if planned else STOCK_PROFILE),
                     "nCtx": n_ctx_for(ptok := PREFIX_TOKENS_MODEL.get((profile, m["id"]), PREFIX_TOKENS[(profile, tmpl)])),
                     "prefixTokens": ptok,
                     **({"kvBytesPerToken": m["kv_cache_bytes_per_token_f16"]} if m.get("kv_cache_bytes_per_token_f16") else {}),
                     **({"note": m["thinking"].get("caveat")} if isinstance(m.get("thinking"), dict) and m["thinking"].get("caveat") else {})},
             "artifacts": [], "evidence": "docs/research/verify/llm-npu.md"}
        if fam == "qwen3.5" and "ru" not in e["languages"]["codes"]:
            e["languages"]["codes"] = ["ru", "en"]
        for f in m.get("download_files", []):
            quant = f.get("quant", "")
            units = [u for u in ["npu", "gpu", "cpu"] if u in f.get("backends", ["cpu"])]
            # llm-npu.md 3.2 (ADR-009): K-quants ARE HTP-tiled at b12520 (the "CPU fallback" claim was REFUTED), so they
            # claim npu too; preferredUnit stays as is (Q4_0 remains the default artifact) and the benchmark decides.
            claimed = list(units)  # preferredUnit is decided on the research claim, unchanged by the npu addition
            if "k_" in quant.lower() and "npu" not in units:
                units = ["npu"] + units
            extra_kv = kv_mb(e["llm"]["nCtx"], e["llm"].get("kvBytesPerToken")) - kv_mb(4096, e["llm"].get("kvBytesPerToken"))
            art = {"id": f"{m['id']}@{f['path'].removesuffix('.gguf').split('-')[-1].lower()}" + ("" if "qad" not in quant.lower() else "-qad"),
                   "runtime": "llama/gguf", "format": "gguf", "quant": quant, "units": units,
                   "files": [hf_file(f["repo"], f.get("repo_sha", "main"), f["path"], bytes_=f["bytes"], sha256=f.get("sha256"), name="model.gguf")],
                   "layout": {"model": "model.gguf"},
                   # estimates were made at n_ctx 4096: add the KV growth to nCtx (ADR-045)
                   "memoryMb": {u: max(extra_kv + CPU_MB.get(m["id"], m.get("ram_estimate_mb", 1200)), CPU_PEAK_MB.get(m["id"], 0))
                                if u == "cpu" else extra_kv + m.get("ram_estimate_mb", 1200) for u in units}}
            pu = PREFERRED.get(tmpl if fam != "qwen3" else "qwen3")
            if pu in claimed:
                art["preferredUnit"] = pu
            if f.get("default"):
                e["artifacts"].insert(0, art)
            else:
                e["artifacts"].append(art)
        # de-duplicate artifact ids (two repos with the same quant)
        seen = {}
        for a in e["artifacts"]:
            if a["id"] in seen:
                a["id"] = a["id"] + "-" + a["files"][0]["repo"].split("/")[0].lower()
            seen[a["id"]] = 1
        if m["id"] == "lfm2.5-2.6b":
            e["notes"] = "Template always opens <think>; no official way to disable reasoning. Experimental."
        if not planned:
            e["followUps"] = [] if any("npu" in a["units"] for a in e["artifacts"]) else [{"id": "npu-port", "note": "No NPU artifact."}]
        else:
            e["description"] = "Murmur's own cleanup fine-tune. Appears here when training is done."
        out.append(e)
    return out


def main() -> int:
    models = asr_models() + aux_models() + llm_models()
    cat = {
        "schema": 1,
        "catalogVersion": CATALOG_VERSION,
        "generated": "2026-11-20 by core/catalog/tools/build_catalog.py" + (" (--offline)" if OFFLINE else " (network-verified)")
                     + (f"; {len(NPU_CTX_PENDING)} NPU context files pending host, not verified (sha256 from native/qnn-contexts/manifest.json)"
                        if NPU_CTX_PENDING else ""),
        "defaults": {
            "asr": {"ru+en": "parakeet-tdt-0.6b-v3", "en": "parakeet-tdt-0.6b-v3", "ru": "gigaam-v3-e2e-rnnt"},
            "vad": "silero-vad-v5", "ocrDet": "pp-ocrv6-tiny-det", "ocrRec": "pp-ocrv5-cyrillic-mobile-rec",
        },
        "packs": [
            {"id": "recommended", "title": "Speech + LFM3 cleanup",
             "description": "Parakeet speech + LFM3 cleanup + screen reading",
             "items": ["parakeet-tdt-0.6b-v3", "silero-vad-v5", "pp-ocrv6-tiny-det", "pp-ocrv5-cyrillic-mobile-rec", "lfm3-1.7b"]},
            {"id": "light", "title": "Lighter and faster",
             "description": "Parakeet speech + Qwen3.5 0.8B cleanup + screen reading",
             "items": ["parakeet-tdt-0.6b-v3", "silero-vad-v5", "pp-ocrv6-tiny-det", "pp-ocrv5-cyrillic-mobile-rec", "qwen3.5-0.8b"]},
            {"id": "speech-only", "title": "Speech only", "recommended": True, "description": "No cleanup",
             "items": ["parakeet-tdt-0.6b-v3", "silero-vad-v5"]},
        ],
        "models": models,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cat, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_files = sum(len(a["files"]) for m in models for a in m["artifacts"])
    print(f"wrote {OUT.relative_to(REPO)}: {len(models)} models, {n_files} files, offline={OFFLINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
