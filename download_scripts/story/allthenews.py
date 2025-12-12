from datasets import load_dataset
import os

os.makedirs("DATA/story", exist_ok=True)

ds = load_dataset("snapshot-of-huggingface/all-the-news-2-1")
ds.save_to_disk("DATA/story/allthenews")
print("done - allthenews")