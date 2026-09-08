"""Human-readable dataset and pipeline metadata logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data.label_mapping import ID2LABEL, LABEL2ID, TARGET_ID_TO_EMOTIONS

logger = logging.getLogger(__name__)


def _normalize_key(key) -> int:
    return int(key)


def label_counts_named(counts: dict) -> dict[str, int]:
    return {
        ID2LABEL[_normalize_key(class_id)]: int(count)
        for class_id, count in sorted(counts.items(), key=lambda item: _normalize_key(item[0]))
    }


def label_percentages_named(percentages: dict) -> dict[str, float]:
    return {
        ID2LABEL[_normalize_key(class_id)]: float(pct)
        for class_id, pct in sorted(percentages.items(), key=lambda item: _normalize_key(item[0]))
    }


def log_label_schema() -> None:
    logger.info("=== Label Schema (7 production classes) ===")
    for class_id in sorted(TARGET_ID_TO_EMOTIONS):
        emotions = TARGET_ID_TO_EMOTIONS[class_id]
        logger.info("  %d %-40s <- %s", class_id, ID2LABEL[class_id], ", ".join(emotions))


def log_dataset_audit(audit: dict) -> None:
    logger.info("=== Raw Dataset Audit ===")
    logger.info("  Rows: %s", audit.get("rows"))
    logger.info("  Columns: %d", len(audit.get("columns", [])))
    unclear = audit.get("unclear_rate")
    logger.info("  Unclear rate: %s", f"{unclear:.4f}" if unclear is not None else "n/a")
    avg_len = audit.get("avg_text_length")
    logger.info("  Avg text length: %s chars", f"{avg_len:.1f}" if avg_len is not None else "n/a")


def _ensure_named_counts(counts: dict) -> dict[str, int]:
    if not counts:
        return {}
    sample_key = next(iter(counts))
    if isinstance(sample_key, str) and sample_key in LABEL2ID:
        return {str(label): int(count) for label, count in counts.items()}
    return label_counts_named(counts)


def _ensure_named_percentages(percentages: dict) -> dict[str, float]:
    if not percentages:
        return {}
    sample_key = next(iter(percentages))
    if isinstance(sample_key, str) and sample_key in LABEL2ID:
        return {str(label): float(pct) for label, pct in percentages.items()}
    return label_percentages_named(percentages)


def _log_class_distribution(title: str, counts: dict, percentages: dict | None = None) -> None:
    logger.info("=== %s ===", title)
    named_counts = _ensure_named_counts(counts)
    named_pcts = _ensure_named_percentages(percentages) if percentages else {}
    for label, count in named_counts.items():
        if named_pcts:
            logger.info("  %-40s %6d  (%5.2f%%)", label, count, named_pcts.get(label, 0.0))
        else:
            logger.info("  %-40s %6d", label, count)


def _log_split_distribution(title: str, distribution: dict) -> None:
    logger.info("  %s:", title)
    for class_id in sorted(ID2LABEL):
        key = str(class_id)
        pct = float(distribution.get(key, distribution.get(class_id, 0.0)))
        logger.info("    %-40s %5.2f%%", ID2LABEL[class_id], pct)


def _safe_get(d: dict | None, *keys, default=None):
    """Safely traverse nested dict keys."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def log_pipeline_stats(stats: dict) -> None:
    if not stats:
        logger.warning("No pipeline stats available to display.")
        return

    log_label_schema()

    if audit := stats.get("audit"):
        log_dataset_audit(audit)

    if mapping := stats.get("mapping"):
        logger.info("=== After Label Mapping ===")
        logger.info(
            "  %d -> %d rows (dropped %d)",
            mapping["initial_rows"],
            mapping["remaining_rows"],
            mapping["dropped_rows"],
        )
        counts = mapping.get("class_counts_named") or mapping.get("class_counts", {})
        percentages = mapping.get("class_percentages_named") or mapping.get("class_percentages", {})
        _log_class_distribution("Mapped Class Counts", counts, percentages)
        if audit_block := mapping.get("mapping_audit"):
            logger.info("=== Mapping Audit (schema %s) ===", audit_block.get("schema_version", "?"))
            logger.info("  NaN label rate: %.2f%%", _safe_get(audit_block, "nan_label_rate", default=0) * 100)
            logger.info("  Multi-label conflict rows: %s", audit_block.get("multi_label_conflict_rows"))
            logger.info("  Unmapped columns: %s", audit_block.get("unmapped_emotion_columns", []))

    if cleaning := stats.get("cleaning"):
        logger.info("=== After Cleaning ===")
        logger.info(
            "  %d -> %d rows (dedup dropped %d, length filter dropped %d)",
            cleaning["initial_rows"],
            cleaning["remaining_rows"],
            cleaning.get("dedup_dropped", 0),
            cleaning.get("length_filter_dropped", 0),
        )

    if split := stats.get("split"):
        logger.info("=== Split Sizes ===")
        logger.info(
            "  train=%d  val=%d  test=%d",
            split["train_size"],
            split["val_size"],
            split["test_size"],
        )
        _log_split_distribution("Train distribution", split["train_distribution"])
        _log_split_distribution("Val distribution", split["val_distribution"])
        _log_split_distribution("Test distribution", split["test_distribution"])

    if leakage := stats.get("leakage"):
        logger.info("=== Leakage Check ===")
        logger.info("  Passed: %s", leakage.get("no_leakage"))

    if balance := stats.get("balance"):
        logger.info("=== Split Class Balance ===")
        logger.info(
            "  Within tolerance (%.1f%%): %s",
            balance.get("tolerance_pct", 0.5),
            balance.get("within_tolerance"),
        )

    if train_balance := stats.get("train_balance"):
        strategy = train_balance.get("strategy", "none")
        logger.info("=== Train Balance (%s) ===", strategy)
        if strategy != "none":
            logger.info(
                "  Rows: %d -> %d",
                _safe_get(train_balance, "rows_before", default=0),
                _safe_get(train_balance, "rows_after", default=0),
            )
            after = train_balance.get("after", {})
            after_counts = after.get("counts_named") or after.get("counts", {})
            after_pcts = after.get("percentages", {})
            _log_class_distribution("Balanced Train Counts", after_counts, after_pcts)

    if "min_train_samples_per_class" in stats:
        logger.info("  Min train samples per class: %d", stats["min_train_samples_per_class"])


def load_stats_from_disk(processed_dir: Path) -> dict | None:
    path = processed_dir / "data_stats.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
