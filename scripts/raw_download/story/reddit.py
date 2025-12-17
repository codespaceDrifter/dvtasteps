from datasets import load_dataset
import os
from scripts.config import home, hfcache

'''
total possible subreddits:
Each split corresponds to a specific subreddit in the following list: "tifu", "explainlikeimfive", "WritingPrompts", "changemyview", "LifeProTips", "todayilearned", "science", "askscience", 
"ifyoulikeblank", "Foodforthought", "IWantToLearn", "bestof", "IAmA", "socialskills", "relationship_advice", "philosophy", "YouShouldKnow", "history", "books", 
"Showerthoughts", "personalfinance", "buildapc", "EatCheapAndHealthy", "boardgames", "malefashionadvice", "femalefashionadvice", "scifi", "Fantasy", 
"Games", "bodyweightfitness", "SkincareAddiction", "podcasts", "suggestmeabook", "AskHistorians", "gaming", "DIY", "mildlyinteresting", "sports", "space", 
"gadgets", "Documentaries", "GetMotivated", "UpliftingNews", "technology", "Fitness", "travel", "lifehacks", "Damnthatsinteresting", "gardening", "programming"
'''

os.makedirs(f"{home}/DATA/raw/story/reddit", exist_ok=True)

subreddits = ["explainlikeimfive", "WritingPrompts", "changemyview", "LifeProTips", "todayilearned", "askscience", 
"IAmA", "relationship_advice", "philosophy", "YouShouldKnow", "Showerthoughts", "personalfinance", "scifi", "AskHistorians", "UpliftingNews",
"Damnthatsinteresting", "programming"]

ds = load_dataset("HuggingFaceGECLM/REDDIT_comments", cache_dir=hfcache)

for sub in subreddits:
    print(f"downloading {sub}...")
    split = ds[sub].filter(lambda x: int(x["score"]) >= 5)
    split.save_to_disk(f"{home}/DATA/raw/story/reddit/{sub.lower()}")
    print(f"done - {sub}")