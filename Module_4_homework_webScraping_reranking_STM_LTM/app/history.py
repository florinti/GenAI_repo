# ---- app/history.py ----
import os, json
from app import config

def load_history() -> dict:
    if os.path.exists(config.HISTORY_FILE):
        return json.load(open(config.HISTORY_FILE))
    return {"topics": []}

def save_history(history: dict):
    with open(config.HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)