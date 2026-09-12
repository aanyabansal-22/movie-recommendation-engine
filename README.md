# MovieLens Recommendation Engine

A reproducible content-based movie recommender that provides 3–5 (or up to 20) meaningful film picks. It uses MovieLens `latest-small` data: genres are converted into TF-IDF features, cosine similarity identifies related movies, and a Bayesian rating adjustment re-ranks them so a handful of ratings cannot overwhelm quality signals.

## Features

- Downloads and caches the public MovieLens data on the first run. If downloading is unavailable, creates a valid bundled 20-movie offline demo catalog so every command still works.
- Cleans and validates data, including missing rating statistics.
- Handles exact and unambiguous partial title searches with useful error messages.
- Uses TF-IDF genre vectors + cosine similarity for transparent content-based recommendations.
- Re-ranks by a Bayesian audience-quality score and excludes movies with fewer than 20 ratings.
- Has a cold-start mode for users who have not rated or selected anything yet.
- Includes an EDA script producing `eda_summary.png`.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python recommender.py "Toy Story" --count 5
python recommender.py --popular --count 5
python eda.py
```

## Methodology

1. **Data**: MovieLens `latest-small`, downloaded from GroupLens and cached in `data/`.
2. **Preprocessing**: join movies with aggregate rating statistics; replace genre separators with tokens; normalize titles for user search.
3. **Similarity**: TF-IDF represents each movie’s genre profile. Cosine similarity identifies films that share genres without privileging longer genre lists.
4. **Re-ranking**: `0.75 × similarity + 0.25 × Bayesian rating / 5`. The Bayesian rating shrinks titles with few ratings toward the overall mean. Movies with fewer than 20 ratings are excluded.
5. **Cold start**: `--popular` ranks frequently rated films using a stronger Bayesian prior.

The approach is intentionally interpretable. Its limitation is that it uses metadata rather than individual viewing histories. A future collaborative-filtering version could learn latent user/movie factors from the `ratings.csv` user IDs.

## Example

```text
python recommender.py "Toy Story" --count 5
```

Each result reports genre, average rating, rating count, genre similarity, and final score so the ranking can be inspected rather than treated as a black box.

## Project files

- `recommender.py` — model, dataset loading, and command-line interface.
- `eda.py` — repeatable exploratory analysis chart.
- `requirements.txt` — Python dependencies.

MovieLens data is not committed: it is automatically downloaded into `data/` on first use. If your network blocks it, the project writes a deterministic local demo dataset into `data/`; delete that folder and rerun when your connection is available to replace it with the full dataset.
