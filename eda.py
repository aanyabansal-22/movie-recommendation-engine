"""Generate a compact EDA chart for the MovieLens recommendation project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recommender import ensure_data


def main() -> None:
    movies, ratings = ensure_data()
    output = Path(__file__).parent / "eda_summary.png"
    source = (Path(__file__).parent / "data" / "source.txt").read_text(encoding="utf-8").strip()
    dataset_label = "MovieLens latest-small" if source == "movielens-latest-small" else "Offline demo catalog"
    genre_counts = movies.assign(genre=movies.genres.str.split("|")).explode("genre")
    genre_counts = genre_counts[genre_counts.genre != "(no genres listed)"]["genre"].value_counts().head(12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(ratings.rating, bins=[0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25], color="#2864dc", edgecolor="white")
    axes[0].set(title="MovieLens rating distribution", xlabel="Rating", ylabel="Number of ratings")
    genre_counts.sort_values().plot.barh(ax=axes[1], color="#e56a54")
    axes[1].set(title="Most represented genres", xlabel="Number of movies", ylabel="")
    fig.suptitle(f"{dataset_label}: exploratory data analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
