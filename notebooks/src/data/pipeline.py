"""End-to-end data engineering pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.data.balance import balance_train_df
from src.data.clean import clean_dataframe
from src.data.label_mapping import apply_label_mapping, label_map_payload
from src.data.load import audit_dataframe, load_raw_dataframe, try_load_official_splits
from src.data.multi_label_mapping import apply_multi_label_mapping
from src.data.reporting import load_stats_from_disk, log_pipeline_stats
from src.data.split import check_class_balance, check_leakage, official_split, stratified_split
from src.paths import CONFIG_DIR, PROCESSED_DIR, ensure_artifact_dirs


def load_config() -> dict:
    config_path = CONFIG_DIR / "train_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def export_splits(
    train_df,
    val_df,
    test_df,
    stats: dict,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map_payload(), handle, indent=2)

    with (output_dir / "data_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return output_dir


PROCESSED_SPLIT_FILES = ("train.csv", "validation.csv", "test.csv", "label_map.json")


def processed_splits_exist(processed_dir: Path | None = None) -> bool:
    processed_dir = processed_dir or PROCESSED_DIR
    return all((processed_dir / name).is_file() for name in PROCESSED_SPLIT_FILES)


def load_processed_splits(processed_dir: Path | None = None):
    import pandas as pd

    processed_dir = processed_dir or PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in PROCESSED_SPLIT_FILES if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Processed splits missing in {processed_dir}: {', '.join(missing)}. "
            "Run run_data_pipeline() or scripts/run_pipeline.py --stage data."
        )

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    return train_df, val_df, test_df


def load_or_build_processed_splits(config: dict | None = None, *, force_rebuild: bool = False) -> tuple:
    ensure_artifact_dirs()
    config = config or load_config()

    if not force_rebuild and processed_splits_exist():
        train_df, val_df, test_df = load_processed_splits()
        cached_stats = load_stats_from_disk(PROCESSED_DIR) or {}
        return train_df, val_df, test_df, cached_stats

    result = run_data_pipeline(config)
    return result["train_df"], result["val_df"], result["test_df"], result["stats"]


def _map_and_clean(raw_df, config: dict) -> tuple:
    track = config.get("track", "singlelabel")
    dedup_policy = config.get("dedup_policy", "consensus")

    if track == "multilabel":
        mapped_df, mapping_log = apply_multi_label_mapping(raw_df, track=track)
    else:
        mapped_df, mapping_log = apply_label_mapping(raw_df)
        mapped_df, _ = apply_multi_label_mapping(mapped_df, track="multilabel")
        before = len(mapped_df)
        mapped_df = mapped_df[mapped_df["n_active_macros"] == 1].reset_index(drop=True)
        mapping_log["singlelabel_rows"] = len(mapped_df)
        mapping_log["singlelabel_dropped"] = before - len(mapped_df)

    cleaned_df, cleaning_log = clean_dataframe(
        mapped_df,
        min_char_length=config.get("min_char_length", 3),
        max_token_length_approx=config.get("max_token_length_approx", 128),
        dedup_policy=dedup_policy,
    )
    return cleaned_df, mapping_log, cleaning_log


def _process_official_splits(raw_splits: dict, config: dict) -> tuple:
    from src.data.label_mapping import apply_label_mapping
    from src.data.multi_label_mapping import apply_multi_label_mapping

    track = config.get("track", "singlelabel")
    dedup_policy = config.get("dedup_policy", "consensus")

    processed = {}
    logs = {}
    for name, df in raw_splits.items():
        if track == "multilabel":
            mapped, log = apply_multi_label_mapping(df, track=track)
        else:
            mapped, log = apply_label_mapping(df)
            mapped, _ = apply_multi_label_mapping(mapped, track="multilabel")
            mapped = mapped[mapped["n_active_macros"] == 1].reset_index(drop=True)
        cleaned, clean_log = clean_dataframe(mapped, dedup_policy=dedup_policy)
        processed[name] = cleaned
        logs[name] = {"mapping": log, "cleaning": clean_log}

    train_df, val_df, test_df, split_log = official_split(
        processed["train"],
        processed["validation"],
        processed["test"],
        dedup_policy=dedup_policy,
    )
    return train_df, val_df, test_df, split_log, logs


def run_data_pipeline(config: dict | None = None) -> dict:
    ensure_artifact_dirs()
    config = config or load_config()
    split_mode = config.get("split_mode", "stratified")
    dedup_policy = config.get("dedup_policy", "consensus")

    mapping_log = {}
    cleaning_log = {}
    split_log = {}
    official_logs = {}

    audit = {}
    if split_mode == "official":
        raw_splits = try_load_official_splits()
        if raw_splits is None:
            print("WARNING: Official TSV splits not found; falling back to stratified split.")
            split_mode = "stratified"
        else:
            audit = {
                "split_mode": "official",
                "train_rows": len(raw_splits["train"]),
                "val_rows": len(raw_splits["validation"]),
                "test_rows": len(raw_splits["test"]),
            }
            train_df, val_df, test_df, split_log, official_logs = _process_official_splits(
                raw_splits, config
            )
            mapping_log = official_logs.get("train", {}).get("mapping", {})
            cleaning_log = official_logs.get("train", {}).get("cleaning", {})

    if split_mode == "stratified":
        raw_df = load_raw_dataframe()
        audit = audit_dataframe(raw_df)
        cleaned_df, mapping_log, cleaning_log = _map_and_clean(raw_df, config)
        train_df, val_df, test_df, split_log = stratified_split(
            cleaned_df,
            random_seed=config.get("random_seed", 42),
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=config.get("val_ratio", 0.1),
            dedup_policy=dedup_policy,
        )

    stats_audit = audit

    balance_strategy = config.get("balance_strategy", "none")
    if balance_strategy != "none":
        train_df, train_balance_log = balance_train_df(
            train_df,
            strategy=balance_strategy,
            target_neutral_pct=config.get("target_neutral_pct", 0.27),
            random_seed=config.get("balance_random_seed", config.get("random_seed", 42)),
        )
    else:
        train_balance_log = {"strategy": "none"}

    leakage = check_leakage(train_df, val_df, test_df)
    balance = check_class_balance(split_log)

    min_train_per_class = train_df["encoded_label"].value_counts().min()
    stats = {
        "audit": stats_audit,
        "mapping": mapping_log,
        "cleaning": cleaning_log,
        "split": split_log,
        "split_mode": split_mode,
        "dedup_policy": dedup_policy,
        "track": config.get("track", "singlelabel"),
        "leakage": leakage,
        "balance": balance,
        "train_balance": train_balance_log,
        "min_train_samples_per_class": int(min_train_per_class),
    }

    output_dir = export_splits(train_df, val_df, test_df, stats)
    log_pipeline_stats(stats)
    print(f"Exported processed data to {output_dir}")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "stats": stats,
        "output_dir": str(output_dir),
    }
