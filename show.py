from pathlib import Path

IGNORE = {
    "__pycache__",
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "benchmark_results",
    "datasets",
    "image_datasets",
    "libs",
}

def show_tree(path: Path, prefix=""):
    entries = [
        p for p in path.iterdir()
        if p.name not in IGNORE
    ]

    entries.sort(key=lambda p: (p.is_file(), p.name.lower()))

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            show_tree(entry, prefix + extension)

if __name__ == "__main__":
    show_tree(Path("."))