# Movie Recommendation Engine

This is a friendly movie recommendation engine made using Python. It helps you find your next movie by suggesting films similar to the one you already like.

For example, if you enter `Toy Story`, the program recommends movies such as `Coco`, `Shrek`, and `Finding Nemo` because they have similar genres. Great choice - there is always another movie waiting for you!

## What this project does

- Takes a movie name from the user.
- Finds movies with similar genres.
- Shows the top 3 to 5 movie recommendations.
- Gives every recommendation a short, spoiler-free reason to watch it.
- Also shows popular movies for a new user who has not selected a movie.
- Creates a simple EDA graph for ratings and movie genres.
- Includes a simple interactive **Movie Night Finder** mode.

## Dataset

The project uses a local movie dataset with popular English and Indian movies such as Toy Story, Avengers, The Matrix, 3 Idiots, Dangal, Baahubali 2, and K.G.F Chapter 2.

The program can also download the MovieLens dataset automatically when internet access is available. The local dataset is included so the project runs even without internet.

## How the recommendation works

This is a **content-based recommendation system**.

1. Every movie has one or more genres, such as Action, Comedy, Drama, or Sci-Fi.
2. The program converts genres into numbers using TF-IDF.
3. It compares movies using cosine similarity.
4. Movies with more similar genres get a higher similarity score.
5. The final result also considers average rating and number of ratings, so low-rated movies do not appear at the top.

## Files

- `recommender.py` - main recommendation program.
- `eda.py` - creates the EDA chart.
- `requirements.txt` - required Python libraries.
- `data/movies.csv` - movie names and genres.
- `data/ratings.csv` - ratings used by the program.
- `eda_summary.png` - EDA output graph.

## How to run

Open the project folder in VS Code and run these commands in the terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run recommendations for a movie:

```powershell
python recommender.py "Toy Story" --count 5
```

Try another movie:

```powershell
python recommender.py "The Matrix" --count 5
```

Show popular movies:

```powershell
python recommender.py --popular --count 5
```

Create the EDA chart:

```powershell
python eda.py
```

Try the interactive Movie Night Finder:

```powershell
python recommender.py --explore
```

## Sample output

For `Toy Story`, the project recommends movies like Coco, Monsters Inc., Shrek, Finding Nemo, and Moana.

## Future improvement

In the future, this project can use collaborative filtering. That would recommend movies based on ratings given by users with similar interests.
