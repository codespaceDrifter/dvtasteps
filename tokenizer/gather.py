# gathers text from Arrow datasets for tokenizer training
import os
from datasets import load_from_disk
from scripts.config import home

RAW_DIR = f"{home}/DATA/raw"

# map of known text field names
TEXT_FIELDS = ["text", "content", "body", "article"]


def get_text_field(ds):
    """returns the text field name for a dataset"""
    sample = ds[0] if not hasattr(ds, 'keys') else ds[list(ds.keys())[0]][0]

    for field in TEXT_FIELDS:
        if field in sample:
            return field

    raise ValueError(f"unknown text field, available: {list(sample.keys())}")


def iter_texts(ds):
    """yields text strings from a dataset (handles DatasetDict or Dataset)"""
    field = get_text_field(ds)

    # DatasetDict has .keys(), Dataset doesn't
    if hasattr(ds, 'keys'):
        for split in ds.keys():
            for item in ds[split]:
                if item[field]:
                    yield item[field]
    else:
        for item in ds:
            if item[field]:
                yield item[field]


def get_all_dataset_paths():
    """returns all Arrow dataset paths in DATA/raw"""
    paths = []
    for root, dirs, files in os.walk(RAW_DIR):
        if "dataset_info.json" in files:
            paths.append(root)
    return paths


def iter_all_texts(per_dataset=100_000):
    """yields text from all datasets, limited per dataset for tokenizer training"""
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
