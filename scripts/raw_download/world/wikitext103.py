from datasets import load_dataset
import os
from scripts.config import home, hfcache

os.makedirs(f"{home}/DATA/raw/world", exist_ok=True)

ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=hfcache)
ds.save_to_disk(f"{home}/DATA/raw/world/wikitext103")

print("done - wikitext103")
