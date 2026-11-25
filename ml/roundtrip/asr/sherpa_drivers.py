"""sherpa-onnx CPU drivers: the same ONNX graphs the phone runs in CPU mode.

Recogniser kinds (``AsrSpec.extra["kind"]``):

- ``offline_transducer`` — NeMo transducers (Parakeet TDT v3, Parakeet Ultra, GigaAM v3 RNN-T).
- ``offline_nemo_ctc``  — GigaAM v3 CTC.
- ``online_transducer`` — Nemotron 3.5 streaming (cache-aware, per-stream ``language`` option) and the
  English-only Nemotron Speech Streaming (no language option).
- ``online_t_one``      — T-one streaming CTC, 8 kHz input (we resample, see ``_transcribe``).
- ``offline_qwen3``     — Qwen3-ASR 0.6B (audio LLM, auto language).
- ``offline_whisper``   — Whisper / Distil-Whisper (language fixed per recogniser instance).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .audio import SAMPLE_RATE, resample
from .base import AsrDriver, AsrSpec, AsrUnavailable
from .fetch import FetchError, HfFile, fetch_hf

# Silence appended to streaming input so the last chunk is flushed (Nemotron chunk = 560 ms).
ONLINE_TAIL_S = 1.2
TONE_LEFT_PAD_S = 0.3  # from the upstream t-one example


def _import_sherpa():
    try:
        import sherpa_onnx
    except ImportError as exc:  # pragma: no cover - environment problem
        raise AsrUnavailable("sherpa-onnx is not installed in the round-trip venv") from exc
    return sherpa_onnx


class SherpaDriver(AsrDriver):
    """Shared plumbing: lazy download, load/unload, per-kind recogniser construction."""

    def __init__(self, spec: AsrSpec, num_threads: int = 2) -> None:
        super().__init__(spec, num_threads)
        self._rec: Dict[str, Any] = {}  # whisper keeps one recogniser per forced language
        self._dir: Optional[Path] = None

    # ---------------------------------------------------------------- files
    def model_dir(self) -> Path:
        if self._dir is None:
            files = [HfFile(**f) for f in self.spec.extra["files"]]
            try:
                self._dir = fetch_hf(files)
            except FetchError as exc:
                raise AsrUnavailable(str(exc)) from exc
        return self._dir

    def _p(self, key: str) -> str:
        return str(self.model_dir() / self.spec.extra["paths"][key])

    # ---------------------------------------------------------------- lifecycle
    @property
    def loaded(self) -> bool:
        return bool(self._rec)

    def load(self) -> None:
        if not self._rec:
            default = "ru" if "ru" in self.spec.languages else "en"
            self._recogniser(self.spec.extra.get("default_lang", default))

    def unload(self) -> None:
        self._rec.clear()

    def _recogniser(self, lang: str):
        key = lang if self.spec.extra["kind"] == "offline_whisper" else "_"
        rec = self._rec.get(key)
        if rec is None:
            # Whisper fixes the language per recogniser; keep only one loaded (two copies of
            # whisper-medium measured 4.5 GB). The runner groups shards by language hint.
            if self._rec:
                self._rec.clear()
                import gc

                gc.collect()
            rec = self._build(lang)
            self._rec[key] = rec
        return rec

    def _build(self, lang: str):
        so = _import_sherpa()
        kind = self.spec.extra["kind"]
        t = self.num_threads
        try:
            if kind == "offline_transducer":
                return so.OfflineRecognizer.from_transducer(
                    encoder=self._p("encoder"), decoder=self._p("decoder"), joiner=self._p("joiner"),
                    tokens=self._p("tokens"), num_threads=t, model_type="nemo_transducer",
                    **self.spec.extra.get("kwargs", {}))
            if kind == "offline_nemo_ctc":
                return so.OfflineRecognizer.from_nemo_ctc(
                    model=self._p("model"), tokens=self._p("tokens"), num_threads=t,
                    **self.spec.extra.get("kwargs", {}))
            if kind == "online_transducer":
                return so.OnlineRecognizer.from_transducer(
                    tokens=self._p("tokens"), encoder=self._p("encoder"), decoder=self._p("decoder"),
                    joiner=self._p("joiner"), num_threads=t, **self.spec.extra.get("kwargs", {}))
            if kind == "online_t_one":
                return so.OnlineRecognizer.from_t_one_ctc(
                    model=self._p("model"), tokens=self._p("tokens"), num_threads=t)
            if kind == "offline_qwen3":
                return so.OfflineRecognizer.from_qwen3_asr(
                    conv_frontend=self._p("conv_frontend"), encoder=self._p("encoder"),
                    decoder=self._p("decoder"), tokenizer=self._p("tokenizer"), num_threads=t,
                    max_new_tokens=256)
            if kind == "offline_whisper":
                return so.OfflineRecognizer.from_whisper(
                    encoder=self._p("encoder"), decoder=self._p("decoder"), tokens=self._p("tokens"),
                    language=lang, task="transcribe", num_threads=t)
        except AsrUnavailable:
            raise
        except Exception as exc:  # sherpa raises RuntimeError/ValueError on bad graphs
            raise AsrUnavailable(f"{self.spec.name}: sherpa-onnx failed to load: {exc}") from exc
        raise AsrUnavailable(f"{self.spec.name}: unknown recogniser kind {kind!r}")

    # ---------------------------------------------------------------- inference
    def _whisper_lang(self, lang_hint: str) -> str:
        if lang_hint in ("ru", "en", "he"):
            return lang_hint
        return ""  # sherpa-onnx whisper: empty language = auto-detect

    def _transcribe(self, samples: np.ndarray, lang_hint: str) -> str:
        kind = self.spec.extra["kind"]
        if kind == "offline_whisper":
            lang = "en" if self.spec.languages == frozenset({"en"}) else self._whisper_lang(lang_hint)
            rec = self._recogniser(lang)
        else:
            rec = self._recogniser("_")
        if kind.startswith("offline"):
            stream = rec.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            rec.decode_stream(stream)
            return stream.result.text
        if kind == "online_t_one":
            sr = 8000
            x = resample(samples, SAMPLE_RATE, sr)
            stream = rec.create_stream()
            stream.accept_waveform(sr, np.zeros(int(TONE_LEFT_PAD_S * sr), dtype=np.float32))
            stream.accept_waveform(sr, x)
            stream.accept_waveform(sr, np.zeros(int(ONLINE_TAIL_S * sr), dtype=np.float32))
        else:  # online_transducer (Nemotron)
            sr = SAMPLE_RATE
            stream = rec.create_stream()
            if len(self.spec.languages) > 1:  # the English-only Nemotron has no language prompt
                hint = lang_hint if lang_hint in ("ru", "en") else "auto"
                stream.set_option("language", hint)
            stream.accept_waveform(sr, samples)
            stream.accept_waveform(sr, np.zeros(int(ONLINE_TAIL_S * sr), dtype=np.float32))
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        return rec.get_result(stream)
