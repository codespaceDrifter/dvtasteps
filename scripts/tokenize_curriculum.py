# tokenizes each dataset separately into its own .bin file for curriculum learning
# replaces tokenize_pretrain.py and tokenize_pretrain_arrow_to_bin.py

import os
import numpy as np
from datasets import load_from_disk
from tokenizer.tokenizer import ByteTokenizer
from tokenizer.gather import iter_texts
from scripts.config import home, TRAINING_CHUNK_SIZE

RAW_DIR = f"{home}/DATA/raw"
OUT_DIR = f"{home}/DATA/tokenized/curriculum"

# HARD CODED START VALUE. START TOKENIZER AT THIS DATASET (ASSUMING PREVIOUSLY INTERRUPTED)
START_AT = 5  # 5 = world/wikipedia

# matches the hard coded names in each download script
DATASETS = [
    # story
    ("story", "tinystories"),
    ("story", "allthenews"),
    ("story", "bookcorpus"),
    ("story", "reddit"),
    # world
    ("world", "wikitext103"),
    ("world", "wikipedia"),
    ("world", "openwebtext2"),
    # math
    ("math", "openwebmath"),
    # code
    ("code", "python"),
    ("code", "cpp"),
    ("code", "javascript"),
    ("code", "html"),
    ("code", "css"),
]


def get_all_subpaths(path):
    """find all dataset paths under a directory (walks into train/, validation/, subreddit folders, etc)"""
    paths = []
    for root, dirs, files in os.walk(path):
        if "dataset_info.json" in files:
            paths.append(root)
    return paths if paths else [path]


def tokenize_to_bin(raw_path, out_path, tok):
    """tokenize a dataset, streaming directly to .bin file"""
    subpaths = get_all_subpaths(raw_path)
    print(f"  found {len(subpaths)} subpath(s)")

    f = open(out_path, 'wb')
    doc_count = 0
    token_count = 0

    total_docs = 0
    for subpath in subpaths:
        ds = load_from_disk(subpath)
        total_docs += len(ds)

    for subpath in subpaths:
        print(f"  loading: {subpath}")
        ds = load_from_disk(subpath)

        for text in iter_texts(ds):
            # (1) -> (seq_len)
            tokens = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
            np.array(tokens, dtype=np.int32).tofile(f)
            token_count += len(tokens)
            doc_count += 1

            # progress
            if doc_count % 100_000 == 0:
                pct = doc_count / total_docs * 100
                print(f"    {doc_count:,}/{total_docs:,} docs ({pct:.1f}%), {token_count:,} tokens")

    # pad to make divisible by TRAINING_CHUNK_SIZE
    remainder = token_count % TRAINING_CHUNK_SIZE
    if remainder:
        pad_count = TRAINING_CHUNK_SIZE - remainder
        np.array([tok.pad_id] * pad_count, dtype=np.int32).tofile(f)
        token_count += pad_count

    f.close()

    size_gb = token_count * 4 / 1e9
    print(f"  total: {doc_count:,} docs, {token_count:,} tokens")
    print(f"  saved: {out_path} ({size_gb:.2f} GB)")



def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = ByteTokenizer.load()

    print(f"output dir: {OUT_DIR}")
    print(f"chunk size: {TRAINING_CHUNK_SIZE}")
    print(f"datasets: {len(DATASETS)} (starting at {START_AT})")
    print()

    for folder, name in DATASETS[START_AT:]:
        raw_path = f"{RAW_DIR}/{folder}/{name}"
        out_dir = f"{OUT_DIR}/{folder}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{name}.bin"

        print(f"=== {name} ===")
        print(f"  raw: {raw_path}")
        assert os.path.exists(raw_path), f"path not found: {raw_path}"

        tokenize_to_bin(raw_path, out_path, tok)
        print()

    print(f"=== DONE ===")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
