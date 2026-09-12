"""A small, reproducible MovieLens recommendation engine.

The engine uses movie genres to build a content-based similarity model, then
combines it with MovieLens rating data for quality-aware recommendations.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).parent / "data"
MIN_VOTES = 20

# A deliberately small, local catalog keeps the project demonstrable in offline
# classrooms and on networks that block downloads. The normal path still uses
# the complete MovieLens dataset whenever it is available.
DEMO_MOVIES = [
    (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy", 4.05),
    (2, "Shrek (2001)", "Adventure|Animation|Children|Comedy|Fantasy", 3.88),
    (3, "Finding Nemo (2003)", "Adventure|Animation|Children|Comedy", 4.00),
    (4, "The Incredibles (2004)", "Action|Adventure|Animation|Children|Comedy", 4.08),
    (5, "The Lion King (1994)", "Adventure|Animation|Children|Drama|Musical", 4.18),
    (6, "Frozen (2013)", "Adventure|Animation|Children|Comedy|Fantasy|Musical", 3.75),
    (7, "Moana (2016)", "Adventure|Animation|Children|Comedy|Fantasy|Musical", 3.92),
    (8, "Coco (2017)", "Adventure|Animation|Children|Comedy|Fantasy", 4.12),
    (9, "Monsters, Inc. (2001)", "Adventure|Animation|Children|Comedy|Fantasy", 3.96),
    (10, "Inside Out (2015)", "Adventure|Animation|Children|Comedy|Drama", 4.10),
    (11, "Avengers: Endgame (2019)", "Action|Adventure|Sci-Fi", 4.15),
    (12, "Avengers: Infinity War (2018)", "Action|Adventure|Sci-Fi", 4.08),
    (13, "Iron Man (2008)", "Action|Adventure|Sci-Fi", 3.95),
    (14, "Spider-Man: No Way Home (2021)", "Action|Adventure|Sci-Fi", 4.02),
    (15, "The Dark Knight (2008)", "Action|Crime|Drama|Thriller", 4.35),
    (16, "Batman Begins (2005)", "Action|Crime|Drama", 3.92),
    (17, "Joker (2019)", "Crime|Drama|Thriller", 3.98),
    (18, "The Matrix (1999)", "Action|Sci-Fi|Thriller", 4.20),
    (19, "Inception (2010)", "Action|Sci-Fi|Thriller", 4.18),
    (20, "Interstellar (2014)", "Adventure|Drama|Sci-Fi", 4.12),
    (21, "Titanic (1997)", "Drama|Romance", 3.86),
    (22, "Avatar (2009)", "Action|Adventure|Sci-Fi", 3.90),
    (23, "Jurassic Park (1993)", "Action|Adventure|Sci-Fi|Thriller", 4.02),
    (24, "Jumanji: Welcome to the Jungle (2017)", "Action|Adventure|Children|Comedy|Fantasy", 3.62),
    (25, "Mission: Impossible - Fallout (2018)", "Action|Adventure|Thriller", 3.96),
    (26, "Star Wars: Episode IV - A New Hope (1977)", "Action|Adventure|Sci-Fi", 4.25),
    (27, "Star Wars: Episode V - The Empire Strikes Back (1980)", "Action|Adventure|Sci-Fi", 4.30),
    (28, "The Lord of the Rings: The Fellowship of the Ring (2001)", "Adventure|Fantasy", 4.22),
    (29, "The Lord of the Rings: The Return of the King (2003)", "Adventure|Fantasy", 4.30),
    (30, "Harry Potter and the Sorcerer's Stone (2001)", "Adventure|Children|Fantasy", 3.78),
    (31, "Harry Potter and the Deathly Hallows: Part 2 (2011)", "Adventure|Fantasy|Mystery", 4.05),
    (32, "The Godfather (1972)", "Crime|Drama", 4.38),
    (33, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller", 4.22),
    (34, "Forrest Gump (1994)", "Comedy|Drama|Romance", 4.16),
    (35, "K.G.F: Chapter 2 (2022)", "Action|Crime|Drama", 4.32),
    (36, "3 Idiots (2009)", "Comedy|Drama", 4.18),
    (37, "Dangal (2016)", "Drama", 4.02),
    (38, "Baahubali 2: The Conclusion (2017)", "Action|Adventure|Drama|Fantasy", 3.88),
]


@dataclass
class Recommendation:
    title: str
    year: str
    genres: str
    rating: float
    ratings_count: int
    similarity: float
    score: float


class MovieRecommender:
    """Content-based recommender with a rating-quality re-ranking stage."""

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        self.movies = self._prepare_movies(movies, ratings)
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        self.genre_matrix = self.vectorizer.fit_transform(self.movies["genres_text"])
        self.title_lookup = self._build_title_lookup()

    @staticmethod
    def _prepare_movies(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
        movies = movies.copy()
        stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        stats = stats.rename(columns={"mean": "average_rating", "count": "ratings_count"})
        movies = movies.merge(stats, on="movieId", how="left")
        movies["average_rating"] = movies["average_rating"].fillna(0.0)
        movies["ratings_count"] = movies["ratings_count"].fillna(0).astype(int)
        movies["genres_text"] = movies["genres"].str.replace("|", " ", regex=False)
        # Keeps titles and franchises searchable without relying on exact casing.
        movies["normalized_title"] = movies["title"].str.casefold().str.strip()
        movies[["display_title", "year"]] = movies["title"].str.extract(r"^(.*?)(?:\s*\((\d{4})\))?$")
        movies["display_title"] = movies["display_title"].str.strip()
        movies["year"] = movies["year"].fillna("—")
        return movies.reset_index(drop=True)

    def _build_title_lookup(self) -> dict[str, list[int]]:
        lookup: dict[str, list[int]] = {}
        for index, title in self.movies["normalized_title"].items():
            lookup.setdefault(title, []).append(index)
        return lookup

    def find_title(self, query: str) -> tuple[int, str]:
        """Resolve an exact or unambiguous partial title, or raise a helpful error."""
        query = query.casefold().strip()
        if query in self.title_lookup:
            index = self.title_lookup[query][0]
            return index, self.movies.at[index, "title"]

        matches = self.movies[self.movies["normalized_title"].str.contains(query, regex=False)]
        if matches.empty:
            raise ValueError(f"No title matched '{query}'. Try a different spelling.")
        if len(matches) > 1:
            options = ", ".join(matches["title"].head(6))
            raise ValueError(f"'{query}' matches several films: {options}. Please be more specific.")
        index = int(matches.index[0])
        return index, self.movies.at[index, "title"]

    def recommend(self, title: str, count: int = 5) -> tuple[str, list[Recommendation]]:
        if not 3 <= count <= 20:
            raise ValueError("count must be between 3 and 20")
        source_index, resolved_title = self.find_title(title)
        similarities = cosine_similarity(self.genre_matrix[source_index], self.genre_matrix).ravel()
        candidates = self.movies.copy()
        candidates["similarity"] = similarities
        candidates = candidates[candidates.index != source_index]
        candidates = candidates[candidates["ratings_count"] >= MIN_VOTES].copy()

        # A Bayesian rating avoids tiny samples dominating the result.  Similarity
        # has most of the weight, while observed audience quality breaks ties.
        global_mean = self.movies.loc[self.movies["ratings_count"] >= MIN_VOTES, "average_rating"].mean()
        prior_weight = 50
        candidates["bayesian_rating"] = (
            (candidates["ratings_count"] / (candidates["ratings_count"] + prior_weight))
            * candidates["average_rating"]
            + (prior_weight / (candidates["ratings_count"] + prior_weight)) * global_mean
        )
        quality = candidates["bayesian_rating"] / 5.0
        candidates["score"] = 0.75 * candidates["similarity"] + 0.25 * quality
        candidates = candidates.sort_values(
            ["score", "ratings_count"], ascending=[False, False]
        ).head(count)

        recommendations = [
            Recommendation(
                title=row.display_title,
                year=row.year,
                genres=row.genres.replace("|", ", "),
                rating=float(row.average_rating),
                ratings_count=int(row.ratings_count),
                similarity=float(row.similarity),
                score=float(row.score),
            )
            for row in candidates.itertuples()
        ]
        return resolved_title, recommendations

    def popular(self, count: int = 5) -> list[Recommendation]:
        """Return quality-weighted popular movies as a cold-start fallback."""
        if not 3 <= count <= 20:
            raise ValueError("count must be between 3 and 20")
        candidates = self.movies[self.movies["ratings_count"] >= MIN_VOTES].copy()
        global_mean = candidates["average_rating"].mean()
        prior_weight = 100
        candidates["score"] = (
            (candidates["ratings_count"] * candidates["average_rating"] + prior_weight * global_mean)
            / (candidates["ratings_count"] + prior_weight)
        )
        return [
            Recommendation(row.display_title, row.year, row.genres.replace("|", ", "),
                           float(row.average_rating), int(row.ratings_count), 0.0, float(row.score))
            for row in candidates.sort_values("score", ascending=False).head(count).itertuples()
        ]


def write_offline_demo_data(data_dir: Path) -> None:
    """Create deterministic, valid local data when MovieLens cannot be reached."""
    movies = pd.DataFrame(DEMO_MOVIES, columns=["movieId", "title", "genres", "target_rating"])
    rating_rows = []
    # 30 ratings per title (well above MIN_VOTES), with a small fixed spread.
    offsets = (-0.5, -0.25, 0, 0, 0, 0.25, 0.5)
    for movie in movies.itertuples():
        for user_id in range(1, 31):
            rating_rows.append({
                "userId": user_id,
                "movieId": movie.movieId,
                "rating": max(0.5, min(5.0, movie.target_rating + offsets[(user_id - 1) % len(offsets)])),
            })
    movies.drop(columns="target_rating").to_csv(data_dir / "movies.csv", index=False)
    pd.DataFrame(rating_rows).to_csv(data_dir / "ratings.csv", index=False)
    (data_dir / "source.txt").write_text("offline-demo\n", encoding="utf-8")


def ensure_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download MovieLens once, validate expected CSVs, then load them."""
    movies_path, ratings_path = data_dir / "movies.csv", data_dir / "ratings.csv"
    if not (movies_path.exists() and ratings_path.exists()):
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / "ml-latest-small.zip"
        print("Downloading MovieLens latest-small dataset…")
        try:
            urlretrieve(DATA_URL, archive)
            with zipfile.ZipFile(archive) as source:
                with source.open("ml-latest-small/movies.csv") as infile, movies_path.open("wb") as outfile:
                    shutil.copyfileobj(infile, outfile)
                with source.open("ml-latest-small/ratings.csv") as infile, ratings_path.open("wb") as outfile:
                    shutil.copyfileobj(infile, outfile)
            (data_dir / "source.txt").write_text("movielens-latest-small\n", encoding="utf-8")
        except (URLError, zipfile.BadZipFile, KeyError) as error:
            movies_path.unlink(missing_ok=True)
            ratings_path.unlink(missing_ok=True)
            write_offline_demo_data(data_dir)
            print(
                "MovieLens download was unavailable; using the bundled offline demo catalog. "
                "Delete data/ and rerun later to fetch the full dataset."
            )
        finally:
            archive.unlink(missing_ok=True)
    movies, ratings = pd.read_csv(movies_path), pd.read_csv(ratings_path)
    required_movies = {"movieId", "title", "genres"}
    required_ratings = {"userId", "movieId", "rating"}
    if not required_movies.issubset(movies.columns) or not required_ratings.issubset(ratings.columns):
        raise ValueError("Dataset CSVs do not have the expected MovieLens columns.")
    return movies, ratings


def print_results(heading: str, recommendations: list[Recommendation]) -> None:
    print(f"\n{heading}")
    print("=" * len(heading))
    for number, movie in enumerate(recommendations, start=1):
        extra = f"similarity {movie.similarity:.0%}, " if movie.similarity else ""
        print(f"{number}. {movie.title} ({movie.year}) — {movie.genres}")
        print(f"   Rating: {movie.rating:.2f}/5 from {movie.ratings_count:,} ratings ({extra}score {movie.score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies using MovieLens ratings and genre similarity.")
    parser.add_argument("title", nargs="?", help="A movie title to use as the recommendation seed")
    parser.add_argument("--count", type=int, default=5, help="Number of results (3–20; default: 5)")
    parser.add_argument("--popular", action="store_true", help="Show cold-start popular picks instead of similar films")
    args = parser.parse_args()
    movies, ratings = ensure_data()
    recommender = MovieRecommender(movies, ratings)
    if args.popular:
        print_results("Popular picks", recommender.popular(args.count))
    elif args.title:
        resolved_title, recommendations = recommender.recommend(args.title, args.count)
        print_results(f"Because you liked {resolved_title}", recommendations)
    else:
        parser.error("provide a movie title, or use --popular for cold-start picks")


if __name__ == "__main__":
    main()
