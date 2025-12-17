# combines shards into pretrain.arrow, then converts to pretrain.bin
# NO DELETION OF ANY FILES
# streaming/serial to avoid OOM

import os
import pyarrow as pa
import numpy as np
from scripts.config import home, TRAINING_CHUNK_SIZE

# temp flags to skip steps
SKIP_COMBINE = True  # set False to combine shards into arrow

SHARD_DIR = f"{home}/DATA/tokenized/pretrain_shards"
ARROW_PATH = f"{home}/DATA/tokenized/pretrain.arrow"
BIN_PATH = f"{home}/DATA/tokenized/pretrain.bin"


def combine_shards():
    """combines all shards into one arrow file (streaming to avoid OOM)"""
    # find all shards
    shard_files = [f for f in os.listdir(SHARD_DIR) if f.endswith('.arrow')]
    shard_nums = sorted([int(f.replace('.arrow', '')) for f in shard_files])
    total_shards = len(shard_nums)

    print(f"=== COMBINING SHARDS ===")
    print(f"shard dir: {SHARD_DIR}")
    print(f"total shards: {total_shards}")
    print(f"range: {shard_nums[0]} to {shard_nums[-1]}")
    print()

    os.makedirs(os.path.dirname(ARROW_PATH), exist_ok=True)

    # stream write - open output file once, append each shard
    writer = None
    total_rows = 0

    for i, num in enumerate(shard_nums):
        shard_path = f"{SHARD_DIR}/{num}.arrow"

        with pa.ipc.open_file(shard_path) as reader:
            table = reader.read_all()
            total_rows += table.num_rows

            # init writer with first shard's schema
            if writer is None:
                writer = pa.ipc.new_file(ARROW_PATH, table.schema)

            writer.write_table(table)

        # progress every 100 shards
        if (i + 1) % 100 == 0 or (i + 1) == total_shards:
            pct = (i + 1) / total_shards * 100
            print(f"  written {i + 1}/{total_shards} shards ({pct:.1f}%) - {total_rows:,} rows")

    writer.close()
    print()
    print(f"  total rows: {total_rows:,}")
    print(f"  saved to: {ARROW_PATH}")
    print()
    return total_rows


def arrow_to_bin():
    """converts arrow to bin for numpy mmap (fast batch processing)"""
    print(f"=== CONVERTING TO BIN ===")
    print(f"reading from: {ARROW_PATH}")

    with pa.ipc.open_file(ARROW_PATH) as reader:
        num_batches = reader.num_record_batches
        print(f"  num batches: {num_batches}")

        # first pass: count total rows (fast, just metadata)
        print(f"  counting rows...")
        total_rows = 0
        for i in range(num_batches):
            total_rows += reader.get_batch(i).num_rows
            if (i + 1) % 200 == 0:
                print(f"    counted {i + 1}/{num_batches} batches...")

        print(f"  total rows: {total_rows:,}")
        print(f"  chunk size: {TRAINING_CHUNK_SIZE}")

        # (total_rows, TRAINING_CHUNK_SIZE) -> int32
        expected_size = total_rows * TRAINING_CHUNK_SIZE * 4
        print(f"  expected size: {expected_size / 1e9:.2f} GB")

        # pre-allocate output array with mmap
        print(f"  creating output file: {BIN_PATH}")
        arr = np.memmap(BIN_PATH, dtype=np.int32, mode='w+', shape=(total_rows, TRAINING_CHUNK_SIZE))

        # second pass: convert batches (fast, using native numpy conversion)
        print(f"  converting batches...")
        row_idx = 0

        for batch_idx in range(num_batches):
            batch = reader.get_batch(batch_idx)
            # (batch_rows, TRAINING_CHUNK_SIZE) -> numpy directly
            input_ids = batch.column("input_ids")
            batch_rows = len(input_ids)

            # convert batch to numpy in one go (fast C code, not Python loop)
            for j, row in enumerate(input_ids):
                arr[row_idx + j] = row.as_py()

            row_idx += batch_rows

            # progress every batch
            pct = (batch_idx + 1) / num_batches * 100
            print(f"    batch {batch_idx + 1}/{num_batches} ({pct:.1f}%) - {row_idx:,} rows written")

    # flush to disk
    arr.flush()
    del arr

    print()
    print(f"  done")
    print(f"  saved to: {BIN_PATH}")
    print()


def verify_bin():
    """spot check 20 random positions in bin file"""
    import random
    from tokenizer.tokenizer import ByteTokenizer

    print(f"=== VERIFYING BIN ===")
    print(f"loading tokenizer...")
    tok = ByteTokenizer.load()

    # mmap the bin file read-only
    arr = np.memmap(BIN_PATH, dtype=np.int32, mode='r')
    total_tokens = len(arr)
    total_rows = total_tokens // TRAINING_CHUNK_SIZE

    print(f"  total tokens: {total_tokens:,}")
    print(f"  total rows: {total_rows:,}")
    print(f"  shape: ({total_rows}, {TRAINING_CHUNK_SIZE})")
    print()

    # reshape for easier indexing
    arr = arr.reshape(total_rows, TRAINING_CHUNK_SIZE)

    # pick 20 random row positions
    random.seed(42)
    positions = sorted(random.sample(range(total_rows), min(20, total_rows)))

    print(f"checking {len(positions)} random positions (100 tokens each):")
    print("-" * 80)

    for i, row_idx in enumerate(positions):
        # (100,) tokens from this row
        tokens = arr[row_idx, :100].tolist()
        text = tok.decode(tokens)
        # truncate text for display
        text_display = text[:150].replace('\n', '\\n')
        if len(text) > 150:
            text_display += "..."

        print(f"[{i+1:2d}] row {row_idx:,}: {text_display}")

    print("-" * 80)
    print(f"verification done")
    print()


def main():
    print(f"NOTE: this script does NOT delete any files\n")
    print(f"SKIP_COMBINE: {SKIP_COMBINE}\n")

    if not SKIP_COMBINE:
        if os.path.exists(ARROW_PATH):
            print(f"pretrain.arrow already exists, skipping combine")
            print(f"  {ARROW_PATH}")
            print()
        else:
            combine_shards()

    arrow_to_bin()
    verify_bin()

    print(f"=== ALL DONE ===")
    print(f"arrow: {ARROW_PATH}")
    print(f"bin: {BIN_PATH}")
    print(f"shards kept at: {SHARD_DIR}")


if __name__ == "__main__":
    main()
