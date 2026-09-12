# Movie Recommendation System

A content-based movie recommender built with Python, pandas, scikit-learn, and Streamlit. It uses TF-IDF on cleaned title + genre text and cosine similarity to recommend movies that are textually similar to the selected movie.

## Overview
This project processes the MovieLens movie metadata and builds a recommendation engine that suggests the most similar titles to the movie a user selects in the app. The app is interactive and deployed as a lightweight Streamlit web interface.

## Features
- Content-based recommendations using title and genres
- TF-IDF + cosine similarity ranking
- Streamlit web app for interactive recommendation search
- Cached model loading for faster repeated recommendations
- Input validation and graceful empty-result handling
- Simple, beginner-friendly project structure

## Tech Stack
- Python 3.13
- pandas
- NumPy
- scikit-learn
- Streamlit
- Matplotlib
- pytest

## Dataset
This project uses the MovieLens latest-small dataset. The app expects the movies metadata file at:

```text
data/movies.csv
```

The dataset was included in the project and is loaded through the project-relative path system in the data loader.

## Project Structure
```text
movie-recommendation-system/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── movies.csv
├── notebooks/
│   └── movie_recommendation_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── recommender.py
├── tests/
│   └── test_recommender.py
└── .venv/
```

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the app locally
```powershell
streamlit run app.py --server.headless true --server.port 8505
```

Open the browser at:

```text
http://localhost:8505
```

## How it works
1. Load the movie metadata CSV.
2. Clean and normalize the title and genres.
3. Combine the cleaned title and genres into one text field per movie.
4. Fit a TF-IDF vectorizer on that text.
5. Compute pairwise cosine similarity between movies.
6. Recommend the top similar movies to the selected input title.

## Validation
The recommendation logic is checked with lightweight tests to confirm:
- valid recommendations return the correct shape
- unknown movies return an empty DataFrame
- empty titles raise a clear ValueError
- invalid counts raise a clear ValueError

## Deployment notes
This project is compatible with Streamlit Community Cloud and other simple Python hosting setups. For deployment, make sure:
- the project includes the `requirements.txt` file
- the app entry point is `app.py`
- the data file is included in the repo or mounted in the deployment environment

## License
This project is provided under the MIT license.

## Author
Your Name
