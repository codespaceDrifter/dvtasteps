from datasets import load_dataset
import os
from scripts.config import home, hfcache

os.makedirs(f"{home}/DATA/raw/world", exist_ok=True)

# cleaned version - deduped, english only, low quality removed
ds = load_dataset("Geralt-Targaryen/openwebtext2", split="train", cache_dir=hfcache)
ds.save_to_disk(f"{home}/DATA/raw/world/openwebtext2")

print(f"done - openwebtext2 ({len(ds)} rows)")