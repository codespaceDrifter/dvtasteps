# does EVERYTHING. from downloading raw data to make into txt to training tokenizer (optional) to tokenizing the txt to tokenized data to deleting the txt and raw data.  
import shutil

# === DOWNLOAD RAW ===
print("=== DOWNLOADING RAW DATASETS ===")

# story
print("--- story ---")
exec(open("scripts/raw_download/story/tinystories.py").read())
exec(open("scripts/raw_download/story/allthenews.py").read())
exec(open("scripts/raw_download/story/reddit.py").read())
exec(open("scripts/raw_download/story/bookcorpus.py").read())

# world
print("--- world ---")
exec(open("scripts/raw_download/world/wikipedia.py").read())
exec(open("scripts/raw_download/world/openwebtext2.py").read())

# math
print("--- math ---")
exec(open("scripts/raw_download/math/openwebmath.py").read())

# code
print("--- code ---")
exec(open("scripts/raw_download/code/starcoderdata.py").read())

# conversation (TODO)
# exec(open("scripts/raw_download/conversation/slimorca.py").read())
# exec(open("scripts/raw_download/conversation/ultrachat.py").read())

print("=== DONE DOWNLOADING ===")

# === CONVERT TO TXT ===
# TODO

# === TRAIN TOKENIZER ===
# TODO

# === TOKENIZE ALL ===
# TODO

# === CLEANUP ===
# TODO