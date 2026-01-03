# gathers text from Arrow datasets for tokenizer training
import os
from datasets import load_from_disk
from scripts.config import home

RAW_DIR = f"{home}/DATA/raw"

# map of known text field names
TEXT_FIELDS = ["text", "content", "body", "article"]


# finds which column contains the text (datasets use different names like "text", "content", "body")
# returns the field name as a string, e.g. "text"
def get_text_field(ds):
    sample = ds[0] if not hasattr(ds, 'keys') else ds[list(ds.keys())[0]][0]

    for field in TEXT_FIELDS:
        if field in sample:
            return field

    raise ValueError(f"unknown text field, available: {list(sample.keys())}")


# loops through a HF dataset and yields each text string one by one
# example: for text in iter_texts(ds): print(text)
def iter_texts(ds):
    field = get_text_field(ds)

    # DatasetDict has .keys() (multiple splits), Dataset doesn't (single split)
    if hasattr(ds, 'keys'):
        for split in ds.keys():
            for item in ds[split]:
                if item[field]:
                    yield item[field]
    else:
        for item in ds:
            if item[field]:
                yield item[field]


# walks DATA/raw and finds all folders containing a HF dataset (has dataset_info.json)
# returns list of paths like ["/DATA/raw/story/tinystories", "/DATA/raw/code/python", ...]
def get_all_dataset_paths():
    paths = []
    for root, dirs, files in os.walk(RAW_DIR):
        if "dataset_info.json" in files:
            paths.append(root)
    return paths


# yields texts from ALL datasets in DATA/raw, sampling up to per_dataset from each
# used for tokenizer training so it sees variety without loading everything
def iter_all_texts(per_dataset=100_000):
    paths = get_all_dataset_paths()
    print(f"found {len(paths)} datasets, sampling {per_dataset:,} each")

    for path in paths:
        print(f"  {path}")
        ds = load_from_disk(path)

        count = 0
        for text in iter_texts(ds):
            yield text
            count += 1
            if per_dataset and count >= per_dataset:
                break

        print(f"    -> {count:,} texts")
