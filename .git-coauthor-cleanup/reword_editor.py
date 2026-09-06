"""Remove Cursor co-author trailer from commit message during rebase."""
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    lines = handle.readlines()

filtered = [
    line
    for line in lines
    if "Co-authored-by: Cursor <cursoragent@cursor.com>" not in line
]

with open(path, "w", encoding="utf-8", newline="") as handle:
    handle.writelines(filtered)
