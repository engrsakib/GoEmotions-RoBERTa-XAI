"""Mark target commits for reword during interactive rebase."""
import sys

path = sys.argv[1]
targets = {
    "96d2990",
    "16b5d61",
}

with open(path, encoding="utf-8") as handle:
    lines = handle.readlines()

out = []
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("pick ") and any(stripped.split()[1].startswith(t) for t in targets):
        out.append(line.replace("pick ", "reword ", 1))
    else:
        out.append(line)

with open(path, "w", encoding="utf-8", newline="") as handle:
    handle.writelines(out)
