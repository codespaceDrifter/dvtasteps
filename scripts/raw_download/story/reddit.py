import shutil
from datasets import load_dataset
import os

subreddits = ["WritingPrompts", "explainlikeimfive", "todayilearned", "philosophy", "IAmA"]

for sub in subreddits:
    print(f"downloading {sub}...")
    ds = load_dataset("HuggingFaceGECLM/REDDIT_comments", split=sub)
    ds = ds.sort("score", reverse=True)
    ds = ds.select(range(min(2_000_000, len(ds))))
    ds.save_to_disk(f"DATA/raw/story/reddit/{sub.lower()}")
    
    # clear cache for this dataset
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/HuggingFaceGECLM___reddit_comments")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    print(f"done - {sub}")