from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_path(relative_path: str) -> Path:
    """Return a project-relative path that works locally and in deployed apps."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_csv(relative_path: str) -> pd.DataFrame:
    """Load a CSV file using a project-relative path and fail clearly if missing."""
    file_path = resolve_data_path(relative_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)


def load_movies() -> pd.DataFrame:
    """Load the MovieLens movies dataset from the project data directory."""
    return load_csv("data/movies.csv")


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary of the dataset for validation and debugging."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "nulls": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample": df.head(3).to_dict(orient="records"),
    }


def inspect_movies() -> dict[str, Any]:
    """Load the movies data and return a validation summary for the project."""
    df = load_movies()
    summary = summarize_dataframe(df)

    required_columns = {"movieId", "title", "genres"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return summary


if __name__ == "__main__":
    summary = inspect_movies()
    print(summary)
