from datasets import load_dataset
import os

os.makedirs("DATA/story/reddit", exist_ok=True)

subreddits = [
    "WritingPrompts",
    "explainlikeimfive",
    "todayilearned",
    "philosophy",
    "IAmA"
]

for sub in subreddits:
    print(f"downloading {sub}...")
    ds = load_dataset("HuggingFaceGECLM/REDDIT_comments", split=sub)
    ds = ds.sort("score", reverse=True)
    ds = ds.select(range(min(2_000_000, len(ds))))  # ~2gb each
    ds.save_to_disk(f"DATA/story/reddit/{sub.lower()}")
    print(f"done - {sub} ({len(ds)} rows)")

print("done - all reddit")