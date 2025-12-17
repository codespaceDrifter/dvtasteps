from datasets import load_dataset
import os
from scripts.config import home, hfcache

os.makedirs(f"{home}/DATA/raw/world", exist_ok=True)

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", cache_dir=hfcache)
ds.save_to_disk(f"{home}/DATA/raw/world/wikipedia")

print(f"done - wikipedia ({len(ds)} rows)")