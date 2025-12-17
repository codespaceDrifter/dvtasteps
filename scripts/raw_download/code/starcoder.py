from datasets import load_dataset
import os
from scripts.config import home, hfcache

os.makedirs(f"{home}/DATA/raw/code", exist_ok=True)

languages = [
    "python",
    "cpp",
    "javascript",
    "html",
    "css",
]

for lang in languages:
    print(f"downloading {lang}...")
    ds = load_dataset("bigcode/starcoderdata", data_dir=lang, split="train", cache_dir=hfcache)
    ds = ds.filter(lambda x: x["max_stars_count"] is not None and x["max_stars_count"] >= 5)
    ds.save_to_disk(f"{home}/DATA/raw/code/{lang}")
    print(f"done - {lang} ({len(ds)} rows)")

print("done - all code")