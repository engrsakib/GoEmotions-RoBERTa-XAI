"""End-to-end data engineering pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.data.clean import clean_dataframe
from src.data.label_mapping import apply_label_mapping, label_map_payload
from src.data.load import audit_dataframe, load_raw_dataframe
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
        "min_train_samples_per_class": int(min_train_per_class),
    }

    output_dir = export_splits(train_df, val_df, test_df, stats)
    print(f"Exported processed data to {output_dir}")
    print(f"Leakage check passed: {leakage['no_leakage']}")
    print(f"Class balance within tolerance: {balance['within_tolerance']}")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "stats": stats,
        "output_dir": str(output_dir),
    }


def load_processed_splits(processed_dir: Path | None = None):
    import pandas as pd

    processed_dir = processed_dir or PROCESSED_DIR
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    return train_df, val_df, test_df
