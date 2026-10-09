from pathlib import Path


tests_dir = Path("tests")

for file_path in tests_dir.rglob("*.py"):
    raw = file_path.read_bytes()

    if b"\x00" not in raw:
        print(f"OK: {file_path}")
        continue

    try:
        content = file_path.read_text(encoding="utf-16")
        file_path.write_text(content, encoding="utf-8")

        print(f"Converted: {file_path}")

    except Exception as e:
        print(f"Failed: {file_path}: {e}")