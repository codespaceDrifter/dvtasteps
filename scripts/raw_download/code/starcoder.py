from datasets import load_dataset
import os

os.makedirs("DATA/code", exist_ok=True)

languages = [
    "python",
    "c++",
    "javascript",
    "html",
    "css",
]

for lang in languages:
    print(f"downloading {lang}...")
    ds = load_dataset("bigcode/starcoderdata", data_dir=lang, split="train")
    ds.save_to_disk(f"DATA/code/{lang}")
    print(f"done - {lang} ({len(ds)} rows)")

print("done - all code")