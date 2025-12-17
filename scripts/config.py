import os

config = {
    "home": "/media/drift/Extreme SSD",
    "hfcache": "/media/drift/Extreme SSD/hfcache",
}

home = config["home"]
hfcache = config["hfcache"]

# set HF cache before any HF imports
os.environ["HF_HOME"] = hfcache

# tokenization
TRAINING_CHUNK_SIZE = 2048
