from datasets import load_dataset
import os

os.makedirs("DATA/world", exist_ok=True)

# cleaned version - deduped, english only, low quality removed
ds = load_dataset("Geralt-Targaryen/openwebtext2", split="train")
ds.save_to_disk("DATA/world/openwebtext2")
print(f"done - openwebtext2 ({len(ds)} rows)")