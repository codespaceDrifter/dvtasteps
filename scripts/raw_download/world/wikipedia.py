from datasets import load_dataset
import os

os.makedirs("DATA/world", exist_ok=True)

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
ds.save_to_disk("DATA/world/wikipedia")
print(f"done - wikipedia ({len(ds)} rows)")