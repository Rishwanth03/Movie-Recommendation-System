from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_movie_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "movies.csv")


def summarize_movie_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return basic counts that help us understand the movie catalog."""
    return {
        "total_movies": int(df.shape[0]),
        "total_unique_movie_ids": int(df["movieId"].nunique()),
        "total_null_titles": int(df["title"].isna().sum()),
        "total_null_genres": int(df["genres"].isna().sum()),
    }


def plot_genre_distribution(df: pd.DataFrame) -> None:
    """Create a simple genre count plot for EDA."""
    genre_series = df["genres"].fillna("").str.split("|").explode()
    genre_series = genre_series[genre_series != ""]

    counts = genre_series.value_counts().head(10)
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="steelblue")
    plt.title("Top Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def analyze_ratings() -> pd.DataFrame | None:
    """Load ratings.csv if it exists and return a basic rating summary."""
    ratings_path = PROJECT_ROOT / "data" / "ratings.csv"
    if not ratings_path.exists():
        return None

    ratings = pd.read_csv(ratings_path)
    print("Ratings shape:", ratings.shape)
    print(ratings["rating"].describe())
    return ratings


def main() -> None:
    df = load_movie_data()
    print("Movie summary:")
    print(summarize_movie_counts(df))

    print("\nTop genres:")
    genre_series = df["genres"].fillna("").str.split("|").explode()
    genre_series = genre_series[genre_series != ""]
    print(genre_series.value_counts().head(10))

    if (PROJECT_ROOT / "data" / "ratings.csv").exists():
        ratings = analyze_ratings()
        if ratings is not None:
            plt.figure(figsize=(8, 5))
            ratings["rating"].plot(kind="hist", bins=20, edgecolor="black")
            plt.title("Rating Distribution")
            plt.xlabel("Rating")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
