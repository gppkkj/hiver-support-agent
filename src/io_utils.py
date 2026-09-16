"""Small, dependency-free helpers for validating CSV inputs."""

import csv
from pathlib import Path
from typing import Iterable, Mapping


def read_csv_rows(path: str | Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    """Read a CSV file and raise a useful error for missing or malformed input."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {csv_path}")
            missing = set(required_columns) - set(reader.fieldnames)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"CSV file {csv_path} is missing columns: {names}")
            rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"Could not parse CSV file {csv_path}: {exc}") from exc

    return [dict(row) for row in rows]


def require_rows(rows: list[Mapping[str, str]], description: str) -> None:
    """Reject empty datasets before downstream code produces confusing output."""
    if not rows:
        raise ValueError(f"{description} is empty; provide at least one row")
