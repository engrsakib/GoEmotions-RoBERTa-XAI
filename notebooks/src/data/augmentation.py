"""Train-set text augmentation for multilabel minority classes (Track A)."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.label_mapping import ID2LABEL, NUM_LABELS
from src.data.multi_label_mapping import str_to_multi_hot
from src.paths import NOTEBOOKS_DIR

logger = logging.getLogger(__name__)

MARIAN_MODELS = {
    "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
    "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
}


def parse_multi_hot_column(series: pd.Series) -> np.ndarray:
    """Parse multi_hot_labels column to (N, C) int array."""
    vectors = [str_to_multi_hot(v) for v in series.tolist()]
    return np.array(vectors, dtype=np.int32)


def is_minority_row(multi_hot: np.ndarray, minority_ids: list[int]) -> bool:
    return any(int(multi_hot[i]) == 1 for i in minority_ids if i < len(multi_hot))


def _text_cache_key(text: str, method: str, lang: str = "") -> str:
    payload = f"{method}|{lang}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BackTranslationAugmenter:
    """English -> foreign -> English via Marian translation pipelines."""

    def __init__(self, langs: list[str] | None = None, cache_dir: Path | None = None):
        self.langs = langs or ["de", "fr"]
        self.cache_dir = cache_dir
        self._pipelines: dict[str, object] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_pair(self, lang: str):
        if lang not in MARIAN_MODELS:
            return None, None
        from transformers import pipeline

        en_to_x, x_to_en = MARIAN_MODELS[lang]
        if en_to_x not in self._pipelines:
            self._pipelines[en_to_x] = pipeline("translation", model=en_to_x)
        if x_to_en not in self._pipelines:
            self._pipelines[x_to_en] = pipeline("translation", model=x_to_en)
        return self._pipelines[en_to_x], self._pipelines[x_to_en]

    def _read_cache(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, key: str, text: str) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.txt"
        path.write_text(text, encoding="utf-8")

    def augment(self, text: str, rng: random.Random) -> str | None:
        lang = rng.choice(self.langs)
        cache_key = _text_cache_key(text, "backtranslation", lang)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        try:
            en_to_x, x_to_en = self._load_pair(lang)
            if en_to_x is None or x_to_en is None:
                return None
            mid = en_to_x(text, max_length=512)[0]["translation_text"]
            out = x_to_en(mid, max_length=512)[0]["translation_text"]
            if out and out.strip() and out.strip().lower() != text.strip().lower():
                self._write_cache(cache_key, out)
                return out.strip()
        except Exception as exc:
            logger.debug("Back-translation failed: %s", exc)
        return None


class SynonymAugmenter:
    """Synonym / contextual word replacement (nlpaug when available)."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir
        self._aug = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_aug(self):
        if self._aug is not None:
            return self._aug
        try:
            import nlpaug.augmenter.word as naw

            self._aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.15)
            return self._aug
        except Exception:
            return None

    def _read_cache(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, key: str, text: str) -> None:
        if not self.cache_dir:
            return
        (self.cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")

    def augment(self, text: str, rng: random.Random) -> str | None:
        cache_key = _text_cache_key(text, "synonym")
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        aug = self._get_aug()
        try:
            if aug is not None:
                out = aug.augment(text)
                if isinstance(out, list):
                    out = out[0]
            else:
                words = text.split()
                if len(words) < 3:
                    return None
                idx = rng.randrange(len(words))
                words[idx] = words[idx]
                out = " ".join(words)
            if out and str(out).strip() and str(out).strip().lower() != text.strip().lower():
                result = str(out).strip()
                self._write_cache(cache_key, result)
                return result
        except Exception as exc:
            logger.debug("Synonym augmentation failed: %s", exc)
        return None


def _default_aug_config(config: dict) -> dict:
    aug = config.get("augmentation") or {}
    cache_rel = aug.get("cache_dir", "artifacts/cache/augmentation")
    cache_path = Path(cache_rel)
    if not cache_path.is_absolute():
        cache_path = (NOTEBOOKS_DIR / cache_rel).resolve()
    return {
        "enabled": bool(aug.get("enabled", False)),
        "minority_macro_ids": list(aug.get("minority_macro_ids", [4, 5, 6])),
        "target_multiplier": float(aug.get("target_multiplier", 2.0)),
        "backtranslation_langs": list(aug.get("backtranslation_langs", ["de", "fr"])),
        "backtranslation_ratio": float(aug.get("backtranslation_ratio", 0.5)),
        "max_augment_per_row": int(aug.get("max_augment_per_row", 1)),
        "random_seed": int(aug.get("random_seed", config.get("random_seed", 42))),
        "cache_dir": cache_path,
    }


def augment_train_dataframe(
    train_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Duplicate minority-class rows via back-translation and synonym replacement.

    Labels (`multi_hot_labels`) are copied unchanged; only `text` is augmented.
    """
    aug_cfg = _default_aug_config(config)
    log: dict = {"enabled": aug_cfg["enabled"], "strategy": "minority_backtranslation_synonym"}

    if not aug_cfg["enabled"]:
        log["skipped"] = "disabled"
        return train_df, log

    if config.get("track", "singlelabel") != "multilabel":
        log["skipped"] = "track_not_multilabel"
        return train_df, log

    if "multi_hot_labels" not in train_df.columns:
        log["skipped"] = "missing_multi_hot_labels"
        return train_df, log

    minority_ids = aug_cfg["minority_macro_ids"]
    multiplier = max(1.0, aug_cfg["target_multiplier"])
    n_aug_per_row = min(
        aug_cfg["max_augment_per_row"],
        max(0, int(np.ceil(multiplier - 1.0))),
    )
    if n_aug_per_row == 0:
        log["skipped"] = "multiplier_le_1"
        return train_df, log

    rng = random.Random(aug_cfg["random_seed"])
    backtrans = BackTranslationAugmenter(
        langs=aug_cfg["backtranslation_langs"],
        cache_dir=aug_cfg["cache_dir"],
    )
    synonym = SynonymAugmenter(cache_dir=aug_cfg["cache_dir"])

    existing_texts = set(train_df["text"].astype(str).str.strip().str.lower())
    multi_hot = parse_multi_hot_column(train_df["multi_hot_labels"])

    new_rows: list[dict] = []
    stats = {
        "rows_before": len(train_df),
        "eligible_rows": 0,
        "backtranslation_ok": 0,
        "backtranslation_fail": 0,
        "synonym_ok": 0,
        "synonym_fail": 0,
        "dedupe_skipped": 0,
    }

    for i in range(len(train_df)):
        row = train_df.iloc[i]
        mh = multi_hot[i]
        if not is_minority_row(mh, minority_ids):
            continue
        stats["eligible_rows"] += 1
        text = str(row["text"])

        for _ in range(n_aug_per_row):
            use_backtrans = rng.random() < aug_cfg["backtranslation_ratio"]
            aug_text = None
            if use_backtrans:
                aug_text = backtrans.augment(text, rng)
                if aug_text:
                    stats["backtranslation_ok"] += 1
                else:
                    stats["backtranslation_fail"] += 1
            else:
                aug_text = synonym.augment(text, rng)
                if aug_text:
                    stats["synonym_ok"] += 1
                else:
                    stats["synonym_fail"] += 1

            if not aug_text:
                continue
            norm = aug_text.strip().lower()
            if norm in existing_texts:
                stats["dedupe_skipped"] += 1
                continue
            existing_texts.add(norm)
            new_row = row.to_dict()
            new_row["text"] = aug_text
            new_rows.append(new_row)

    if not new_rows:
        log.update(stats)
        log["rows_after"] = len(train_df)
        log["rows_added"] = 0
        return train_df, log

    augmented = pd.DataFrame(new_rows)
    result = pd.concat([train_df, augmented], ignore_index=True)
    rng.shuffle(result.index.to_list())
    result = result.sample(frac=1.0, random_state=aug_cfg["random_seed"]).reset_index(drop=True)

    log.update(stats)
    log["rows_after"] = len(result)
    log["rows_added"] = len(new_rows)
    log["minority_macros"] = {ID2LABEL[i]: i for i in minority_ids if i in ID2LABEL}

    print(
        f"Augmentation: +{len(new_rows)} rows ({stats['eligible_rows']} eligible minority rows), "
        f"total train={len(result)}"
    )
    return result, log
