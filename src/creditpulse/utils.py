"""Shared helpers: config, paths, seeding, run tracker, transforms."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_hash"] = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    return cfg


def p(cfg: dict, key: str) -> Path:
    out = ROOT / cfg["paths"][key]
    out.mkdir(parents=True, exist_ok=True)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)  # CPU kernels are deterministic for a fixed thread count
    except ImportError:
        pass


def signed_log(x):
    """C8: sign-preserving log transform, keeps negative balances meaningful."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        info["commit"] = "no-git"
    return info


class Tracker:
    """Lightweight local experiment tracker (MLflow-compatible fields, JSONL storage).

    Every run logs config hash, data version, environment, params and metrics.
    Test-set evaluations are registered separately so R-45 (one test evaluation
    per final model) can be audited.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = p(cfg, "reports")
        self.runs = self.dir / "runs.jsonl"
        self.test_log = self.dir / "test_evaluations.jsonl"
        dv = p(cfg, "interim") / "data_version.txt"
        self.data_version = dv.read_text().strip() if dv.exists() else "unknown"

    def log(self, name: str, params: dict, metrics: dict, kind: str = "run") -> str:
        rid = hashlib.sha1(f"{name}{time.time()}{random.random()}".encode()).hexdigest()[:8]
        rec = {
            "run_id": rid,
            "name": name,
            "kind": kind,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_hash": self.cfg["_config_hash"],
            "data_version": self.data_version,
            "env": env_info(),
            "params": params,
            "metrics": metrics,
        }
        target = self.test_log if kind == "test" else self.runs
        with open(target, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        return rid
