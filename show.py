from pathlib import Path

def count_lines(directory):
    total = 0
    for path in Path(directory).rglob("*.py"):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            total += sum(1 for _ in f)
    return total

print(count_lines("src"))