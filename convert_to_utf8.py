from pathlib import Path


app_dir = Path("app")

for file_path in app_dir.rglob("*.py"):
    try:
        content = file_path.read_text(encoding="utf-16")
        file_path.write_text(content, encoding="utf-8")

        print(f"Converted: {file_path}")

    except Exception as e:
        print(f"Skipped/failed: {file_path}: {e}")