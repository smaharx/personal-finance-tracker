import json
import os

MEMORY_FILE = "data/ai_memory.json"

def load_memory():
    """Loads the AI's categorized keywords from memory, or creates default ones."""
    default_memory = {
        "Food": ["pizza", "burger", "kfc", "restaurant", "groceries", "walmart", "mcdonalds", "coffee"],
        "Transport": ["uber", "gas", "shell", "lyft", "transit", "subway"],
        "Utilities": ["electric", "water", "internet", "wifi", "trash"],
        "Housing": ["rent", "mortgage", "hoa", "maintenance"],
        "Shopping": ["amazon", "target", "clothes", "electronics"],
        "Entertainment": ["netflix", "movie", "steam", "spotify", "hulu"]
    }

    print("Booting up AI Memory Core...")

    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                memory = json.load(f)
                print("Successfully loaded AI's past memory.")
                return memory
        except json.JSONDecodeError:
            print("Warning: Memory file is corrupted! Reverting to default brain.")
            return default_memory
        except Exception as e:
            print(f"Error loading memory: {e}. Reverting to default.")
            return default_memory
    else:
        print("First time boot detected! Starting with a fresh default brain.")
        # Automatically save the default memory so it exists for next time
        save_memory(default_memory)
        return default_memory


def save_memory(memory):
    """Saves the AI's updated keywords safely to the hard drive."""
    # Safety Check: Make sure the 'data' folder actually exists before saving!
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        print(f"Critical Error: Could not save AI memory. {e}")