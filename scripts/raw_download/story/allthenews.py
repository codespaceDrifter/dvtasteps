from datasets import load_dataset
import os

os.makedirs("DATA/raw/story", exist_ok=True)

ds = load_dataset("rjac/all-the-news-2-1-Component-one")
ds.save_to_disk("DATA/raw/story/allthenews")
print("done - allthenews")
