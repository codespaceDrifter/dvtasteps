from datasets import load_dataset
import os

os.makedirs("DATA/math", exist_ok=True)

ds = load_dataset("open-web-math/open-web-math", split="train")
ds.save_to_disk("DATA/math/openwebmath")
print(f"done - openwebmath ({len(ds)} rows)")