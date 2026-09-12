"""Simple content-based movie recommendation engine."""

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FOLDER = Path(__file__).parent / "data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# This small list lets the project work even if MovieLens cannot be downloaded.
DEMO_MOVIES = [
    ("Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy", 4.05),
    ("Shrek (2001)", "Adventure|Animation|Children|Comedy|Fantasy", 3.88),
    ("Finding Nemo (2003)", "Adventure|Animation|Children|Comedy", 4.00),
    ("The Incredibles (2004)", "Action|Adventure|Animation|Children|Comedy", 4.08),
    ("The Lion King (1994)", "Adventure|Animation|Children|Drama|Musical", 4.18),
    ("Frozen (2013)", "Adventure|Animation|Children|Comedy|Fantasy|Musical", 3.75),
    ("Moana (2016)", "Adventure|Animation|Children|Comedy|Fantasy|Musical", 3.92),
    ("Coco (2017)", "Adventure|Animation|Children|Comedy|Fantasy", 4.12),
    ("Monsters, Inc. (2001)", "Adventure|Animation|Children|Comedy|Fantasy", 3.96),
    ("Inside Out (2015)", "Adventure|Animation|Children|Comedy|Drama", 4.10),
    ("Avengers: Endgame (2019)", "Action|Adventure|Sci-Fi", 4.15),
    ("Avengers: Infinity War (2018)", "Action|Adventure|Sci-Fi", 4.08),
    ("Iron Man (2008)", "Action|Adventure|Sci-Fi", 3.95),
    ("Spider-Man: No Way Home (2021)", "Action|Adventure|Sci-Fi", 4.02),
    ("The Dark Knight (2008)", "Action|Crime|Drama|Thriller", 4.35),
    ("The Matrix (1999)", "Action|Sci-Fi|Thriller", 4.20),
    ("Inception (2010)", "Action|Sci-Fi|Thriller", 4.18),
    ("Interstellar (2014)", "Adventure|Drama|Sci-Fi", 4.12),
    ("Titanic (1997)", "Drama|Romance", 3.86),
    ("Jurassic Park (1993)", "Action|Adventure|Sci-Fi|Thriller", 4.02),
    ("Star Wars: Episode V - The Empire Strikes Back (1980)", "Action|Adventure|Sci-Fi", 4.30),
    ("Harry Potter and the Sorcerer's Stone (2001)", "Adventure|Children|Fantasy", 3.78),
    ("The Godfather (1972)", "Crime|Drama", 4.38),
    ("Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller", 4.22),
    ("Forrest Gump (1994)", "Comedy|Drama|Romance", 4.16),
    ("3 Idiots (2009)", "Comedy|Drama", 4.18),
    ("Dangal (2016)", "Drama", 4.02),
    ("Baahubali 2: The Conclusion (2017)", "Action|Adventure|Drama|Fantasy", 3.88),
    ("K.G.F: Chapter 2 (2022)", "Action|Crime|Drama", 4.32),
]


def create_demo_data():
    """Save a small local dataset when download is unavailable."""
    DATA_FOLDER.mkdir(exist_ok=True)
    movies = []
    ratings = []
    changes = [-0.5, -0.25, 0, 0, 0, 0.25, 0.5]

    for movie_id, (title, genres, average) in enumerate(DEMO_MOVIES, start=1):
        movies.append([movie_id, title, genres])
        # Add 30 sample ratings so every movie has enough ratings to compare.
        for user_id in range(1, 31):
            rating = average + changes[(user_id - 1) % len(changes)]
            ratings.append([user_id, movie_id, max(0.5, min(5.0, rating))])

    pd.DataFrame(movies, columns=["movieId", "title", "genres"]).to_csv(
        DATA_FOLDER / "movies.csv", index=False
    )
    pd.DataFrame(ratings, columns=["userId", "movieId", "rating"]).to_csv(
        DATA_FOLDER / "ratings.csv", index=False
    )
    (DATA_FOLDER / "source.txt").write_text("offline-demo\n", encoding="utf-8")


def load_data():
    """Load local data. On first run, try MovieLens and fall back to demo data."""
    movies_file = DATA_FOLDER / "movies.csv"
    ratings_file = DATA_FOLDER / "ratings.csv"

    if not movies_file.exists() or not ratings_file.exists():
        DATA_FOLDER.mkdir(exist_ok=True)
        zip_file = DATA_FOLDER / "movielens.zip"
        try:
            print("Downloading MovieLens dataset...")
            urlretrieve(MOVIELENS_URL, zip_file)
            with zipfile.ZipFile(zip_file) as data_zip:
                for name, output in [("movies.csv", movies_file), ("ratings.csv", ratings_file)]:
                    with data_zip.open(f"ml-latest-small/{name}") as source, output.open("wb") as target:
                        shutil.copyfileobj(source, target)
            (DATA_FOLDER / "source.txt").write_text("movielens\n", encoding="utf-8")
        except (URLError, zipfile.BadZipFile, KeyError):
            print("MovieLens could not be downloaded. Using local demo data instead.")
            create_demo_data()
        finally:
            zip_file.unlink(missing_ok=True)

    return pd.read_csv(movies_file), pd.read_csv(ratings_file)


def prepare_data(movies, ratings):
    """Add the average rating and rating count for every movie."""
    movie_ratings = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    movie_ratings.columns = ["movieId", "average_rating", "rating_count"]
    movies = movies.merge(movie_ratings, on="movieId", how="left")
    movies["average_rating"] = movies["average_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    # Replace | so genres become simple words for TF-IDF.
    movies["genre_words"] = movies["genres"].str.replace("|", " ", regex=False)
    return movies


def recommend(movie_name, number_of_movies=5):
    """Return movies with genres similar to the selected movie."""
    movies, ratings = load_data()
    movies = prepare_data(movies, ratings)

    selected = movies[movies["title"].str.contains(movie_name, case=False, regex=False)]
    if selected.empty:
        print(f"Movie '{movie_name}' was not found. Try another name.")
        return

    selected_index = selected.index[0]
    selected_title = movies.loc[selected_index, "title"]

    vectorizer = TfidfVectorizer()
    genre_matrix = vectorizer.fit_transform(movies["genre_words"])
    similarity_scores = cosine_similarity(genre_matrix[selected_index], genre_matrix).flatten()
    movies["similarity"] = similarity_scores

    results = movies[(movies.index != selected_index) & (movies["rating_count"] >= 20)].copy()
    # Similarity is the main factor. Rating is used to order similar movies better.
    results["final_score"] = 0.8 * results["similarity"] + 0.2 * (results["average_rating"] / 5)
    results = results.sort_values("final_score", ascending=False).head(number_of_movies)

    print(f"\nBecause you liked {selected_title}\n")
    for position, movie in enumerate(results.itertuples(), start=1):
        print(f"{position}. {movie.title} - {movie.genres}")
        print(f"   Rating: {movie.average_rating:.2f}/5 from {movie.rating_count} ratings")


def popular_movies(number_of_movies=5):
    """Show highly rated movies for new users."""
    movies, ratings = load_data()
    movies = prepare_data(movies, ratings)
    results = movies[movies["rating_count"] >= 20].sort_values("average_rating", ascending=False)

    print("\nPopular movies\n")
    for position, movie in enumerate(results.head(number_of_movies).itertuples(), start=1):
        print(f"{position}. {movie.title} - {movie.genres}")
        print(f"   Rating: {movie.average_rating:.2f}/5 from {movie.rating_count} ratings")


def main():
    parser = argparse.ArgumentParser(description="Simple movie recommendation engine")
    parser.add_argument("movie", nargs="?", help="Movie name, for example: Toy Story")
    parser.add_argument("--count", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--popular", action="store_true", help="Show popular movies")
    args = parser.parse_args()

    if args.popular:
        popular_movies(args.count)
    elif args.movie:
        recommend(args.movie, args.count)
    else:
        print('Example: python recommender.py "Toy Story" --count 5')


if __name__ == "__main__":
    main()
