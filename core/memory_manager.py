import json
import os

MEMORY_FILE = "data/ai_memory.json"


def load_memory():

    default_memory = {
        "Food": ["pizza", "burger", "kfc"],
        "Transport": ["uber", "gas"],
        "Utilities": ["electric", "water"],
        "Housing": ["rent"],
        "Shopping": ["amazon"]
    }

    if os.path.exists(MEMORY_FILE):

        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)

        except:
            return default_memory

    return default_memory


def save_memory(memory):

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)