from joblib import dump, load
from pathlib import Path

CACHE_PATH = Path("data/processed")
CACHE_PATH.mkdir(parents=True, exist_ok=True)

def save_cache(obj, name):
    dump(obj, CACHE_PATH/f"{name}.joblib")

def load_cache(name):
    path = CACHE_PATH/ f"{name}.joblib"
    if path.exists():
        return load(path)
    return None


