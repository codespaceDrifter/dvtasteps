from datasets import load_dataset
import os
from scripts.config import home, hfcache

os.makedirs(f"{home}/DATA/raw/math", exist_ok=True)

ds = load_dataset("open-web-math/open-web-math", split="train", cache_dir=hfcache)
ds.save_to_disk(f"{home}/DATA/raw/math/openwebmath")

print(f"done - openwebmath ({len(ds)} rows)")