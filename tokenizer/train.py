# trains ByteTokenizer on all raw datasets
from tokenizer.gather import iter_all_texts, get_all_dataset_paths
from tokenizer.tokenizer import ByteTokenizer

print("=== TRAINING TOKENIZER ===\n")

# show datasets
paths = get_all_dataset_paths()
print(f"datasets ({len(paths)}):")
for p in paths:
    print(f"  {p}")
print()

# train
tok = ByteTokenizer.create()
tok.train(iter_all_texts())
tok.save()

# test
print("\n=== TEST ===")
tok = ByteTokenizer.load()

tests = [
    "Hello, world!",
    "def fib(n):\n\treturn n if n <= 1 else fib(n-1) + fib(n-2)",
]

for text in tests:
    ids = tok.encode(text)
    back = tok.decode(ids)
    print(f"\n'{text}'")
    print(f"  -> {ids}")
    print(f"  -> '{back}' (match: {text == back})")

print(f"\npad={tok.pad_id} unk={tok.unk_id} bos={tok.bos_id} eos={tok.eos_id}")
print(f"vocab_size={len(tok)}")
