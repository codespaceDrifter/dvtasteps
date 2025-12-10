from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")
ds.save_to_disk("../../DATA/story/tinystories")
print("done - tinystories")