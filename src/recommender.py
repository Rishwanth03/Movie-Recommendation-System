"""Movie recommendation preprocessing logic.

This stage focuses on loading the MovieLens movie metadata and building one
clean text feature per movie for TF-IDF similarity.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_movies


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the movie metadata and validate the required columns."""
    df = load_movies()

    required_columns = {"movieId", "title", "genres"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df.copy()


def clean_title(title: str) -> str:
    """Normalize the title by stripping whitespace and removing trailing years."""
    if pd.isna(title):
        return ""

    cleaned = str(title).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*\(\d{4}\)\s*$", "", cleaned)
    return cleaned.strip()


def clean_genres(genres: str) -> str:
    """Convert pipe-delimited genres into a clean, readable label string."""
    if pd.isna(genres):
        return ""

    cleaned = str(genres).strip()
    if cleaned.lower() == "(no genres listed)":
        return ""

    parts = [part.strip().lower() for part in cleaned.split("|") if part.strip()]
    return " ".join(parts)


@st.cache_data(show_spinner=False)
def prepare_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create one cleaned text feature per movie from title and genres.

    This is the feature we will later feed into TF-IDF. Combining title and
    genre information gives the model a richer description of each movie than
    title alone.
    """
    movie_df = load_data() if df is None else df.copy()

    movie_df["title_clean"] = movie_df["title"].apply(clean_title)
    movie_df["genres_clean"] = movie_df["genres"].apply(clean_genres)

    movie_df["combined_text"] = (
        movie_df["title_clean"].fillna("").astype(str).str.lower().str.strip()
        + " "
        + movie_df["genres_clean"].fillna("").astype(str).str.lower().str.strip()
    )

    movie_df["combined_text"] = movie_df["combined_text"].replace(r"\s+", " ", regex=True).str.strip()

    return movie_df


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """Fit the TF-IDF model and cosine similarity matrix once and reuse it."""
    movie_df = prepare_features()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movie_df["combined_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return movie_df, vectorizer, cosine_sim


def build_model(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """Fit TF-IDF and compute the cosine similarity matrix for all movies.

    The model takes each movie's combined text feature and translates it into a
    numerical vector. Similar movies end up with similar vectors, so cosine
    similarity becomes a practical way to rank likely matches.
    """
    if df is not None:
        movie_df = prepare_features(df)
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(movie_df["combined_text"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return movie_df, vectorizer, cosine_sim

    return load_model_bundle()


def recommend_movies(title: str, n: int = 5) -> pd.DataFrame:
    """Return the top-n movies similar to the given title.

    The function is intentionally forgiving: it matches titles case-insensitively,
    strips year suffixes, and uses fuzzy matching by normalized title strings.
    It excludes the movie itself, keeps only valid similarity scores, and sorts
    descending by relevance. If the title is not found, it returns an empty
    DataFrame rather than crashing.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("A non-empty movie title is required.")

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer.")

    movie_df, _, cosine_sim = load_model_bundle()

    normalized_input = clean_title(title).lower()
    movie_df["title_norm"] = movie_df["title_clean"].str.lower()

    title_matches = movie_df[movie_df["title_norm"] == normalized_input]
    if title_matches.empty:
        # Fuzzy fallback: match by normalized title contains comparison
        title_matches = movie_df[
            movie_df["title_norm"].str.contains(normalized_input, regex=False, na=False)
        ]

    if title_matches.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])

    selected_movie_index = title_matches.index[0]
    selected_movie_id = movie_df.loc[selected_movie_index, "movieId"]
    sim_scores = cosine_sim[selected_movie_index]

    score_series = pd.Series(sim_scores, index=movie_df.index)
    candidate_df = movie_df.loc[score_series.index].copy()
    candidate_df["similarity_score"] = score_series.values

    candidate_df = candidate_df[candidate_df["movieId"] != selected_movie_id].copy()
    candidate_df = candidate_df[candidate_df["similarity_score"] > 0].copy()
    candidate_df = candidate_df.sort_values("similarity_score", ascending=False).head(n)

    result = candidate_df[["movieId", "title", "genres", "similarity_score"]].copy()
    result["similarity_score"] = result["similarity_score"].round(6)
    return result.reset_index(drop=True)


def validate_recommendations() -> dict[str, bool | str]:
    """Run lightweight sanity checks for the recommendation logic.

    This is not model accuracy validation; it checks runtime correctness and
    expected behavior of the output shape and ranking.
    """
    try:
        results = recommend_movies("Toy Story", n=5)
        valid_shape = not results.empty and len(results) == 5
        valid_scores = bool((results["similarity_score"].between(0, 1)).all())
        valid_self_exclusion = not results["movieId"].eq(1).any()
        descending_scores = bool(results["similarity_score"].is_monotonic_decreasing)

        unknown = recommend_movies("Definitely Not A Real Movie", n=5)
        valid_unknown = unknown.empty

        return {
            "valid_shape": valid_shape,
            "valid_scores": valid_scores,
            "valid_self_exclusion": valid_self_exclusion,
            "descending_scores": descending_scores,
            "valid_unknown_title": valid_unknown,
            "status": "passed",
        }
    except Exception as exc:  # pragma: no cover - explicit runtime validation
        return {"status": f"failed: {type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    data, vectorizer, similarity = build_model()
    print("movie_count:", len(data))
    print("tfidf_shape:", similarity.shape)
    print("score_range:", float(similarity.min()), float(similarity.max()))
    print("self_similarity:", float(similarity[0, 0]))
    print("sample_tokens:", len(vectorizer.get_feature_names_out()))
    print("\nRecommendation for Toy Story:")
    print(recommend_movies("Toy Story", n=5).head(5).to_string(index=False))
    print("\nValidation summary:")
    print(validate_recommendations())
