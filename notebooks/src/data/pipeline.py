"""End-to-end data engineering pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.data.balance import balance_train_df
from src.data.clean import clean_dataframe
from src.data.label_mapping import apply_label_mapping, label_map_payload
from src.data.load import audit_dataframe, load_raw_dataframe
from src.data.reporting import load_stats_from_disk, log_pipeline_stats
from src.data.split import check_class_balance, check_leakage, stratified_split
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


def run_data_pipeline(config: dict | None = None) -> dict:
    ensure_artifact_dirs()
    config = config or load_config()

    raw_df = load_raw_dataframe()
    audit = audit_dataframe(raw_df)

    mapped_df, mapping_log = apply_label_mapping(raw_df)
    cleaned_df, cleaning_log = clean_dataframe(
        mapped_df,
        min_char_length=config.get("min_char_length", 3),
        max_token_length_approx=config.get("max_token_length_approx", 128),
    )

    train_df, val_df, test_df, split_log = stratified_split(
        cleaned_df,
        random_seed=config.get("random_seed", 42),
        train_ratio=config.get("train_ratio", 0.8),
        val_ratio=config.get("val_ratio", 0.1),
    )

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
        "audit": audit,
        "mapping": mapping_log,
        "cleaning": cleaning_log,
        "split": split_log,
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


