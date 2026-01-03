# downloads all raw datasets to USB
import shutil

print("=== DOWNLOADING RAW DATASETS ===")

# story
print("--- story ---")
#exec(open("scripts/raw_download/story/tinystories.py").read())
#exec(open("scripts/raw_download/story/allthenews.py").read())
#exec(open("scripts/raw_download/story/reddit.py").read())
#exec(open("scripts/raw_download/story/bookcorpus.py").read())

# world
print("--- world ---")
#exec(open("scripts/raw_download/world/wikitext103.py").read())
#exec(open("scripts/raw_download/world/wikipedia.py").read())
#exec(open("scripts/raw_download/world/openwebtext2.py").read())

# math
print("--- math ---")
#exec(open("scripts/raw_download/math/openwebmath.py").read())

# code
print("--- code ---")
exec(open("scripts/raw_download/code/starcoder.py").read())

print("=== DONE DOWNLOADING RAW DATASETS ===")