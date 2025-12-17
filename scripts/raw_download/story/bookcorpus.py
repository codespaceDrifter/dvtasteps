from datasets import load_dataset
import os
from scripts.config import home, hfcache

os.makedirs(f"{home}/DATA/raw/story", exist_ok=True)

# using cleaned version (no script, parquet format)
ds = load_dataset("Yuti/bookcorpus", split="train", cache_dir=hfcache)
ds.save_to_disk(f"{home}/DATA/raw/story/bookcorpus")

print(f"done - bookcorpus ({len(ds)} rows)")