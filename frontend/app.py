import json
import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import streamlit as st

from backend.recommender import load_data, recommend_movies


def get_tmdb_api_key() -> str | None:
    """Return the TMDB API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.getenv("TMDB_API_KEY")


def get_tmdb_poster_url(movie_id: int) -> str | None:
    """Fetch a TMDB poster URL for a movie if the optional configuration exists."""
    links_path = Path("data") / "links.csv"
    if not links_path.exists():
        return None

    try:
        links_df = pd.read_csv(links_path)
    except Exception:
        return None

    if "movieId" not in links_df.columns or "tmdbId" not in links_df.columns:
        return None

    tmdb_id = links_df.loc[links_df["movieId"] == movie_id, "tmdbId"]
    if tmdb_id.empty:
        return None

    api_key = get_tmdb_api_key()
    if not api_key:
        return None

    try:
        tmdb_movie_id = int(tmdb_id.iloc[0])
        url = (
            f"https://api.themoviedb.org/3/movie/{tmdb_movie_id}"
            f"?api_key={api_key}&language=en-US"
        )
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        poster_path = payload.get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")

    st.sidebar.title("Movie Finder")
    st.sidebar.caption("Content-based recommendation demo")

    movies_df = load_data()
    movie_titles = sorted(movies_df["title"].dropna().unique().tolist())

    selected_title = st.sidebar.selectbox(
        "Choose a movie",
        movie_titles,
        index=movie_titles.index("Toy Story (1995)"),
    )
    num_recommendations = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    show_posters = st.sidebar.checkbox("Show optional posters", value=False)

    st.title("🎬 Movie Recommendation System")
    st.caption("Discover movies with similar themes, genres, and titles.")

    st.markdown("---")

    try:
        with st.spinner("Finding similar movies..."):
            results = recommend_movies(selected_title, n=num_recommendations)
    except ValueError as exc:
        st.error(f"Unable to generate recommendations: {exc}")
        results = None

    if results is not None and results.empty:
        st.warning("No matching recommendations were found for that title.")
    elif results is not None:
        st.subheader(f"Recommended for: {selected_title}")

        for index, row in results.iterrows():
            cols = st.columns([1, 4])
            if show_posters:
                poster_url = get_tmdb_poster_url(int(row["movieId"]))
                if poster_url:
                    with cols[0]:
                        st.image(poster_url, width=100)
                else:
                    with cols[0]:
                        st.caption("Poster unavailable")
            with cols[-1]:
                st.markdown(
                    f"### {index + 1}. {row['title']}\n"
                    f"**Genres:** {row['genres']}\n"
                    f"**Similarity:** {row['similarity_score']:.4f}"
                )
            st.divider()

    st.markdown("---")
    st.caption("Built with Python, pandas, scikit-learn, and Streamlit.")


if __name__ == "__main__":
    main()
