# tokenizes ALL raw datasets into one big pretrain file

import os
import pyarrow as pa
from datasets import load_from_disk
from tokenizer.tokenizer import ByteTokenizer
from tokenizer.gather import get_all_dataset_paths, iter_texts
from scripts.config import home, TRAINING_CHUNK_SIZE
OUT_PATH = f"{home}/DATA/tokenized/pretrain.arrow"
SAVE_EVERY = 100_000

def main():
    tok = ByteTokenizer.load()
    paths = get_all_dataset_paths()

    print(f"tokenizing {len(paths)} datasets")
    print(f"output: {OUT_PATH}\n")

    buffer = []
    chunks = []
    doc_count = 0
    last_tokens = None

    for path in paths:
        print(f"{path}")
        ds = load_from_disk(path)

        for text in iter_texts(ds):
            # (doc_count) -> (doc_count + 1)
            tokens = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
            last_tokens = tokens
            buffer.extend(tokens)
            doc_count += 1

            # (buffer) -> (chunks)
            while len(buffer) >= TRAINING_CHUNK_SIZE:
                chunks.append({"input_ids": buffer[:TRAINING_CHUNK_SIZE]})
                buffer = buffer[TRAINING_CHUNK_SIZE:]

            # save every N docs
            if doc_count % SAVE_EVERY == 0 and chunks:
                save_chunks(chunks, doc_count, tok.decode(last_tokens[:200])[:300])
                chunks = []

    # leftover buffer (pad it)
    if buffer:
        buffer.extend([tok.pad_id] * (TRAINING_CHUNK_SIZE - len(buffer)))
        chunks.append({"input_ids": buffer})

    # final save
    if chunks:
        sample = tok.decode(last_tokens[:200])[:300] if last_tokens else ""
        save_chunks(chunks, doc_count, sample)

    print(f"\ndone - {doc_count} docs")
    print(f"shards saved to: {shard_dir}")
    print(f"run tokenize_pretrain_arrow_to_bin.py to combine and convert")


shard_count = 0
shard_dir = OUT_PATH.replace(".arrow", "_shards")

def save_chunks(chunks, doc_count=0, sample=""):
    global shard_count
    table = pa.table({"input_ids": [c["input_ids"] for c in chunks]})
    shard_path = f"{shard_dir}/{shard_count}.arrow"
    os.makedirs(shard_dir, exist_ok=True)

    with pa.ipc.new_file(shard_path, table.schema) as writer:
        writer.write_table(table)

    print(f"  shard {shard_count} saved - {doc_count} docs total")
    if sample:
        print(f"  sample: {sample}...")
    shard_count += 1


if __name__ == "__main__":
    main()
