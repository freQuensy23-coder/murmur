"""YAML run configs with `extends:` inheritance, ${ENV} expansion and dotted CLI overrides."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(v):
    if isinstance(v, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = merge(load(path.parent / parent), cfg)
    return cfg


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    """`train.max_steps=5 method=lora lora.r=8` -> nested keys; values parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for it in items or []:
        key, _, val = it.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve(path: str | Path, overrides: list[str] | None = None) -> dict:
    return _expand(apply_overrides(load(path), overrides or []))


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "base.yaml"


def base_revision(cfg: dict, base: str | None = None) -> str | None:
    """The Hub commit to load `base` (default cfg["base_model"]) at: an explicit `base_revision`, else the
    `base_revisions` table entry for that repo id. A local directory needs none (None). A Hub repo id without
    a pinned revision is an error: the pod must never train on whatever the vendor pushed last."""
    base = base or cfg["base_model"]
    if Path(base).is_dir():
        return None
    rev = cfg.get("base_revision") or (cfg.get("base_revisions") or {}).get(base)
    if not rev:
        raise SystemExit(f"no pinned revision for {base!r}: add it to base_revisions in ml/train/configs/base.yaml "
                         "(full commit sha from the HF API) or pass --set base_revision=<sha>")
    if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit(f"invalid pinned revision for {base!r}: use a full 40-character commit SHA")
    return rev


def pinned_revisions(path: str | Path = DEFAULT_CONFIG) -> dict[str, str]:
    """repo id -> commit sha table of base.yaml (used by ml/pod/bootstrap.sh and export_eval.sh)."""
    return dict(load(path).get("base_revisions") or {})


if __name__ == "__main__":  # `python -m ml.train.config revision REPO` -> prints the pinned sha (bootstrap.sh)
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "revision":
        print(base_revision({"base_revisions": pinned_revisions()}, sys.argv[2]))
    else:
        raise SystemExit("usage: python -m ml.train.config revision REPO_ID")
