from datasets import load_dataset
import os
from scripts.config import home, hfcache

os.makedirs(f"{home}/DATA/raw/story", exist_ok=True)

ds = load_dataset("rjac/all-the-news-2-1-Component-one", cache_dir=hfcache)
ds.save_to_disk(f"{home}/DATA/raw/story/allthenews")

print("done - allthenews")
