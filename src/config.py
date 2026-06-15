from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "ml-32m"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

RATINGS_FILE = RAW_DIR / "ratings.csv"
MOVIES_FILE = RAW_DIR / "movies.csv"
TAGS_FILE = RAW_DIR / "tags.csv"

MOVIES_PROCESSED = PROCESSED_DIR / "movies.parquet"
INTERACTIONS_PROCESSED = PROCESSED_DIR / "interactions_positive.parquet"
USER_MAP_FILE = PROCESSED_DIR / "user_map.parquet"
MOVIE_CF_MAP_FILE = PROCESSED_DIR / "movie_cf_map.parquet"

ALS_MODEL_FILE = MODELS_DIR / "als_model.joblib"
TFIDF_VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"
USER_ITEM_FILE = MODELS_DIR / "user_item.npz"
MOVIE_TFIDF_FILE = MODELS_DIR / "movie_tfidf.npz"

DEFAULT_MIN_RATING = 4.0
DEFAULT_RANDOM_STATE = 42

