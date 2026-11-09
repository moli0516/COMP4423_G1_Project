import csv
import hashlib
import os
import tempfile
from pathlib import Path

from src.exceptions import TrainingError


def read_table(path, required):
    path = Path(path)
    if not path.is_file():
        raise TrainingError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TrainingError(f"{path.name} is empty or has no header.")
        headers = [name.strip() for name in reader.fieldnames]
        if len(headers) != len(set(headers)):
            raise TrainingError(f"{path.name}: duplicate column names.")
        missing = required - set(headers)
        if missing:
            raise TrainingError(f"{path.name} is missing columns: {sorted(missing)}")
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row:
                raise TrainingError(f"{path.name}:{number}: too many columns.")
            cleaned = {key.strip(): (value or "").strip() for key, value in row.items()}
            cleaned["_line"] = number
            rows.append(cleaned)
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image_path(dataset_dir, relative_path):
    root = Path(dataset_dir).resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise TrainingError(f"Image path leaves the dataset directory: {relative_path}") from error
    return path


def write_atomic(path, writer, binary=False):
    """Replace one destination; multiple destinations are not a transaction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
        with tempfile.NamedTemporaryFile(
            "wb" if binary else "w", dir=path.parent, prefix=f".{path.stem}_",
            suffix=".tmp", delete=False, **kwargs,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
