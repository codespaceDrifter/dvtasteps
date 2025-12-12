description for the flow of the code

scripts/raw_download folder download all datasets in raw form to DATA/raw  
scripts/txt_from_raw to make all of raw dataset into individual txts (no train test split) to DATA/txts. this deletes the stuff in DATA/raw  
we run tokenizer_sample.py to gather 100 mb from each dataset and then we run BPE to train the tokenizer to tokenizer/tokenizer_training_data. this involves multiple retokenizing of the txt. after finished the tokenizer/tokenizer_training_data are deleted. save the trained tokenizer into tokenizer/weights  
scripts/tokenize_from_txt to make tokenized dataset from txts to DATA/tokenized. this deletes stuff in DATA/txts
