"""Discover and normalize metrics/history for publication figures and paper generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.label_mapping import ID2LABEL, NUM_LABELS
from src.data.multi_label_mapping import str_to_multi_hot
from src.paths import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    CONFIG_DIR,
    EXPORTS_DIR,
    NOTEBOOKS_DIR,
    PROCESSED_DIR,
)


@dataclass
class TraceEntry:
    key: str
    value: Any
    source: str
    note: str = ""


@dataclass
class PublicationBundle:
    primary_model_id: str = "m6_deberta_v3"
    training_history: list[dict[str, float]] = field(default_factory=list)
    test_metrics_default: dict[str, float] = field(default_factory=dict)
    test_metrics_thresholded: dict[str, float] = field(default_factory=dict)
    val_test_gap: float | None = None
    threshold_sensitivity: list[dict[str, float]] = field(default_factory=list)
    per_class_f1: dict[str, float] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    cooccurrence_matrix: np.ndarray | None = None
    benchmark_rows: list[dict[str, Any]] = field(default_factory=list)
    ablation_rows: list[dict[str, Any]] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)
    sources_scanned: list[str] = field(default_factory=list)
    used_demo_fallback: bool = False


def _trace(bundle: PublicationBundle, key: str, value: Any, source: str, note: str = "") -> None:
    bundle.trace.append(TraceEntry(key=key, value=value, source=source, note=note))


def _normalize_metric(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    return v


def _metrics_from_block(block: dict | None) -> dict[str, float]:
    if not block:
        return {}
    keys = (
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "micro_f1",
        "hamming_loss",
        "subset_accuracy",
    )
    out = {}
    for k in keys:
        if k in block and block[k] is not None:
            if k == "hamming_loss":
                out[k] = float(block[k])
            else:
                nv = _normalize_metric(block[k])
                if nv is not None:
                    out[k] = nv
    return out


def _discover_json_files(*roots: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root or not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name.startswith(".") or "node_modules" in path.parts:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return sorted(found, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _pick_primary_experiment(json_files: list[Path], model_id: str) -> tuple[dict | None, Path | None]:
    priority_names = (
        f"experiment_E2_thresholds.json",
        "experiment_E2.json",
        "training_metrics.json",
    )
    for name in priority_names:
        for path in json_files:
            if path.name == name and path.parent == EXPORTS_DIR:
                payload = _load_json(path)
                if payload:
                    return payload, path
    for path in json_files:
        payload = _load_json(path)
        if not payload:
            continue
        if payload.get("model_id") == model_id or payload.get("experiment_id") == "E2":
            return payload, path
    for path in json_files:
        if "experiment" in path.name.lower():
            payload = _load_json(path)
            if payload and payload.get("track") == "multilabel":
                return payload, path
    return None, None


def _trainer_state_paths(*roots: Path) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root or not root.exists():
            continue
        for name in ("trainer_state.json", "trainer_log_history.json"):
            paths.extend(root.rglob(name))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _history_from_trainer_state(path: Path) -> list[dict[str, float]]:
    payload = _load_json(path)
    if not payload:
        return []
    history = payload.get("log_history") or payload.get("history") or []
    cleaned: list[dict[str, float]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        entry: dict[str, float] = {}
        for k, v in row.items():
            if isinstance(v, (int, float)):
                entry[k] = float(v)
        if entry:
            cleaned.append(entry)
    return cleaned


def _label_distribution_from_processed(bundle: PublicationBundle) -> None:
    train_csv = PROCESSED_DIR / "train.csv"
    stats_json = PROCESSED_DIR / "data_stats.json"
    if stats_json.is_file():
        stats = _load_json(stats_json)
        bundle.sources_scanned.append(str(stats_json))
        mapping = (stats or {}).get("mapping") or {}
        counts = mapping.get("class_counts") or (stats or {}).get("balance", {}).get("counts_named")
        if counts and isinstance(counts, dict):
            bundle.label_distribution = {str(k): int(v) for k, v in counts.items()}
            _trace(bundle, "label_distribution", bundle.label_distribution, str(stats_json))
            return

    if not train_csv.is_file():
        return
    bundle.sources_scanned.append(str(train_csv))
    df = pd.read_csv(train_csv, nrows=50000)
    if "multi_hot_labels" in df.columns:
        vectors = df["multi_hot_labels"].apply(str_to_multi_hot).tolist()
        counts = np.array(vectors).sum(axis=0)
        bundle.label_distribution = {
            ID2LABEL[i]: int(counts[i]) for i in range(min(NUM_LABELS, len(counts)))
        }
    elif "encoded_label" in df.columns:
        vc = df["encoded_label"].value_counts()
        bundle.label_distribution = {
            ID2LABEL.get(int(k), str(k)): int(v) for k, v in vc.items()
        }
    _trace(bundle, "label_distribution", bundle.label_distribution, str(train_csv))


def _cooccurrence_from_processed(bundle: PublicationBundle) -> None:
    train_csv = PROCESSED_DIR / "train.csv"
    if not train_csv.is_file() or "multi_hot_labels" not in pd.read_csv(train_csv, nrows=1).columns:
        return
    df = pd.read_csv(train_csv, usecols=["multi_hot_labels"], nrows=8000)
    mat = np.stack(df["multi_hot_labels"].apply(str_to_multi_hot).tolist()).astype(float)
    if mat.size == 0:
        return
    corr = np.corrcoef(mat.T)
    bundle.cooccurrence_matrix = np.nan_to_num(corr, nan=0.0)
    bundle.sources_scanned.append(str(train_csv))
    _trace(bundle, "cooccurrence_matrix", "7x7", str(train_csv))


def _threshold_sensitivity_from_probs(bundle: PublicationBundle, exports_dir: Path) -> None:
    npz = exports_dir / "ensemble_probs" / f"{bundle.primary_model_id}_val.npz"
    if not npz.is_file():
        return
    data = np.load(npz)
    probs, labels = data["probs"], data["labels"]
    from src.training.metrics import compute_multilabel_metrics

    grid = []
    for t in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= t).astype(int)
        m = compute_multilabel_metrics(labels, preds)
        grid.append({"threshold": float(t), "macro_f1": float(m["macro_f1"])})
    bundle.threshold_sensitivity = grid
    bundle.sources_scanned.append(str(npz))
    _trace(bundle, "threshold_sensitivity", len(grid), str(npz))


def _per_class_f1_from_probs(bundle: PublicationBundle, exports_dir: Path) -> None:
    npz = exports_dir / "ensemble_probs" / f"{bundle.primary_model_id}_test.npz"
    if not npz.is_file():
        return
    from sklearn.metrics import f1_score

    data = np.load(npz)
    probs, labels = data["probs"], data["labels"]
    t = 0.5
    if bundle.test_metrics_thresholded:
        t = 0.5
    preds = (probs >= t).astype(int)
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    bundle.per_class_f1 = {ID2LABEL[i]: float(f1s[i]) for i in range(min(len(f1s), NUM_LABELS))}
    bundle.sources_scanned.append(str(npz))
    _trace(bundle, "per_class_f1", bundle.per_class_f1, str(npz))


def _benchmark_from_exports(json_files: list[Path], bundle: PublicationBundle) -> None:
    rows = []
    for path in json_files:
        if not path.name.startswith("experiment_"):
            continue
        payload = _load_json(path)
        if not payload:
            continue
        mid = payload.get("model_id", path.stem)
        default_m = _metrics_from_block(payload.get("test_metrics_default"))
        tuned_m = _metrics_from_block(payload.get("test_metrics_thresholded") or payload.get("test_metrics"))
        if not default_m and not tuned_m:
            eval_m = payload.get("eval_metrics") or {}
            tuned_m = _metrics_from_block(eval_m)
        rows.append(
            {
                "id": mid,
                "name": mid,
                "macro_f1_default": default_m.get("macro_f1"),
                "macro_f1_tuned": tuned_m.get("macro_f1"),
                "source": str(path),
            }
        )
    if rows:
        bundle.benchmark_rows = rows
        _trace(bundle, "benchmark_rows", len(rows), "experiment_*.json")


def _apply_experiment_payload(bundle: PublicationBundle, payload: dict, source: Path) -> None:
    bundle.sources_scanned.append(str(source))
    bundle.primary_model_id = payload.get("model_id") or bundle.primary_model_id
    bundle.test_metrics_default = _metrics_from_block(payload.get("test_metrics_default")) or bundle.test_metrics_default
    tuned = _metrics_from_block(
        payload.get("test_metrics_thresholded") or payload.get("test_metrics")
    )
    if tuned:
        bundle.test_metrics_thresholded = tuned
    gap = payload.get("val_test_macro_f1_gap")
    if gap is not None:
        bundle.val_test_gap = float(gap)
    tlog = payload.get("threshold_log") or {}
    if isinstance(tlog, dict):
        for key in ("threshold_sweep", "grid", "macro_f1_by_threshold"):
            if key in tlog and isinstance(tlog[key], list):
                bundle.threshold_sensitivity = [
                    {"threshold": float(r.get("threshold", r.get("t", 0.5))), "macro_f1": float(r["macro_f1"])}
                    for r in tlog[key]
                    if isinstance(r, dict) and "macro_f1" in r
                ]
    per_class = payload.get("per_class_f1") or payload.get("per_class_metrics")
    if isinstance(per_class, dict):
        bundle.per_class_f1 = {str(k): float(v) for k, v in per_class.items()}


def _load_demo_fallback(bundle: PublicationBundle) -> None:
    path = CONFIG_DIR / "publication_defaults.yaml"
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    bundle.used_demo_fallback = True
    bundle.sources_scanned.append(str(path))
    bundle.primary_model_id = data.get("primary_model_id", bundle.primary_model_id)
    bundle.test_metrics_default = _metrics_from_block(data.get("headline_test_default")) or bundle.test_metrics_default
    bundle.test_metrics_thresholded = (
        _metrics_from_block(data.get("headline_test_thresholded")) or bundle.test_metrics_thresholded
    )
    if data.get("val_test_macro_f1_gap") is not None:
        bundle.val_test_gap = float(data["val_test_macro_f1_gap"])
    if data.get("benchmark_rows"):
        bundle.benchmark_rows = list(data["benchmark_rows"])
    if data.get("ablation_rows"):
        bundle.ablation_rows = list(data["ablation_rows"])
    if not bundle.threshold_sensitivity:
        best = bundle.test_metrics_thresholded.get("macro_f1", 0.8074)
        base = bundle.test_metrics_default.get("macro_f1", 0.7892)
        bundle.threshold_sensitivity = [
            {"threshold": t, "macro_f1": base + (best - base) * max(0, 1 - abs(t - 0.45) * 2)}
            for t in [round(x * 0.05, 2) for x in range(2, 19)]
        ]
    if not bundle.per_class_f1:
        macro = bundle.test_metrics_thresholded.get("macro_f1", 0.8074)
        bundle.per_class_f1 = {ID2LABEL[i]: macro * (0.85 + 0.03 * (i % 3)) for i in range(NUM_LABELS)}
    if not bundle.training_history:
        bundle.training_history = _synthetic_training_history()
    _trace(bundle, "demo_fallback", True, str(path), "smoke-test defaults only")


def _synthetic_training_history() -> list[dict[str, float]]:
    rows = []
    for epoch in range(1, 6):
        rows.append({"epoch": float(epoch), "loss": 0.45 / epoch + 0.05})
        rows.append(
            {
                "epoch": float(epoch),
                "eval_loss": 0.40 / epoch + 0.04,
                "eval_macro_f1": 0.55 + 0.05 * epoch,
            }
        )
    return rows


def apply_headline_config(bundle: PublicationBundle, config_path: Path) -> None:
    """Override headline test metrics from YAML (Kaggle final numbers for paper traceability)."""
    if not config_path.is_file():
        return
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("headline_test_thresholded"):
        bundle.test_metrics_thresholded = _metrics_from_block(data["headline_test_thresholded"])
    if data.get("headline_test_default"):
        bundle.test_metrics_default = _metrics_from_block(data["headline_test_default"])
    if data.get("val_test_macro_f1_gap") is not None:
        bundle.val_test_gap = float(data["val_test_macro_f1_gap"])
    if data.get("benchmark_rows"):
        bundle.benchmark_rows = list(data["benchmark_rows"])
    if data.get("ablation_rows"):
        bundle.ablation_rows = list(data["ablation_rows"])
    bundle.used_demo_fallback = False
    _trace(bundle, "headline_config_override", str(config_path), str(config_path))


def load_publication_bundle(
    artifacts_dir: Path | None = None,
    kaggle_log_dir: Path | None = None,
    primary_model_id: str = "m6_deberta_v3",
    allow_demo_fallback: bool = True,
    headline_config: Path | None = None,
) -> PublicationBundle:
    """Discover metrics from repo artifacts and optional Kaggle log directory."""
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    exports_dir = artifacts_dir / "exports" if (artifacts_dir / "exports").exists() else EXPORTS_DIR
    if not exports_dir.exists():
        exports_dir = EXPORTS_DIR

    search_roots = [
        exports_dir,
        artifacts_dir,
        CHECKPOINTS_DIR,
        CHECKPOINTS_DIR / primary_model_id,
        PROCESSED_DIR,
    ]
    if kaggle_log_dir:
        search_roots.extend([Path(kaggle_log_dir), Path(kaggle_log_dir) / "exports"])

    bundle = PublicationBundle(primary_model_id=primary_model_id)
    json_files = _discover_json_files(*search_roots)
    payload, src = _pick_primary_experiment(json_files, primary_model_id)
    if payload and src:
        _apply_experiment_payload(bundle, payload, src)

    for ts_path in _trainer_state_paths(*search_roots):
        hist = _history_from_trainer_state(ts_path)
        if hist:
            bundle.training_history = hist
            bundle.sources_scanned.append(str(ts_path))
            _trace(bundle, "training_history", len(hist), str(ts_path))
            break

    _label_distribution_from_processed(bundle)
    _cooccurrence_from_processed(bundle)
    _threshold_sensitivity_from_probs(bundle, exports_dir)
    _per_class_f1_from_probs(bundle, exports_dir)
    if not bundle.benchmark_rows:
        _benchmark_from_exports(json_files, bundle)

    if not bundle.ablation_rows and (CONFIG_DIR / "publication_defaults.yaml").is_file():
        with (CONFIG_DIR / "publication_defaults.yaml").open(encoding="utf-8") as handle:
            demo = yaml.safe_load(handle) or {}
        bundle.ablation_rows = list(demo.get("ablation_rows") or [])

    if headline_config:
        apply_headline_config(bundle, headline_config)

    needs_fallback = not bundle.test_metrics_thresholded.get("macro_f1")
    if needs_fallback and allow_demo_fallback:
        _load_demo_fallback(bundle)

    return bundle


def validate_headline_metrics(bundle: PublicationBundle, strict: bool = True) -> list[str]:
    """Return list of validation errors; empty if OK."""
    errors: list[str] = []
    req = bundle.test_metrics_thresholded
    for key in ("macro_f1", "macro_precision", "macro_recall", "micro_f1"):
        if key not in req:
            errors.append(f"Missing headline test metric: {key}")
    if strict and bundle.used_demo_fallback:
        errors.append(
            "Demo fallback metrics in use; mount Kaggle artifacts or pass real experiment JSON before submission."
        )
    return errors


def traceability_manifest(bundle: PublicationBundle) -> dict[str, Any]:
    return {
        "primary_model_id": bundle.primary_model_id,
        "used_demo_fallback": bundle.used_demo_fallback,
        "sources_scanned": bundle.sources_scanned,
        "headline_test_thresholded": bundle.test_metrics_thresholded,
        "headline_test_default": bundle.test_metrics_default,
        "val_test_gap": bundle.val_test_gap,
        "trace": [
            {"key": t.key, "value": t.value, "source": t.source, "note": t.note} for t in bundle.trace
        ],
    }


def format_percent(proportion: float | None, digits: int = 2) -> str:
    if proportion is None:
        return "N/A"
    return f"{proportion * 100:.{digits}f}"
