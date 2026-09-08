# External Validation (Planned)

Place held-out datasets here for cross-domain evaluation (Paper 14 pattern):

- SemEval-2018 Task 1 samples mapped through schema v2
- Manually labeled Reddit/Twitter snippets (500–1000 rows)

Expected CSV columns: `text`, `encoded_label` (0–6)

Run: `python scripts/run_experiments.py --experiment E3` then evaluate exported model on these files.
