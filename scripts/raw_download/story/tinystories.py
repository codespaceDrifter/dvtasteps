from datasets import load_dataset
import os

os.makedirs("DATA/raw/story", exist_ok=True)

ds = load_dataset("roneneldan/TinyStories")
ds.save_to_disk("DATA/raw/story/tinystories")
print("done - tinystories")