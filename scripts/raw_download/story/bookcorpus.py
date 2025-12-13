from datasets import load_dataset
import os

os.makedirs("DATA/story", exist_ok=True)

# using cleaned version that streams properly
ds = load_dataset("bookcorpus/bookcorpus", split="train")
ds.save_to_disk("DATA/story/bookcorpus")
print(f"done - bookcorpus ({len(ds)} rows)")