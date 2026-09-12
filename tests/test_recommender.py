import pandas as pd
import pytest

from backend.recommender import recommend_movies


def test_recommendations_return_expected_shape():
    results = recommend_movies("Toy Story", n=3)
    assert len(results) == 3
    assert list(results.columns) == ["movieId", "title", "genres", "similarity_score"]
    assert results["similarity_score"].between(0, 1).all()


def test_unknown_title_returns_empty_dataframe():
    results = recommend_movies("Definitely Not A Real Movie", n=5)
    assert results.empty


def test_empty_title_raises_value_error():
    with pytest.raises(ValueError, match="non-empty movie title"):
        recommend_movies("   ")


def test_non_integer_count_raises_value_error():
    with pytest.raises(ValueError, match="positive integer"):
        recommend_movies("Toy Story", n=1.5)
