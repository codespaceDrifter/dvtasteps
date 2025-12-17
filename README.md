# dvtasteps

complete datasets for AI training  
to be used by the expvr repo for different model experiments and primy for aligned agi attempts  
actual data in the DATA folder. includes a custom built tokenizer. tokenized data in TOKENIZED folder.

download_scripts to download web datasets, synthetic_scripts to generate synthetic verifiable datasets, maybe later add rl_envs for enviroments for agents.  

these are the datasets organized in curriculum with both different topics and progression of difficulty within a topic. meant to be used with some curriculum training wrapper in the svnapse repo. meant to be used in the exact order as written.    

# Tokenizer 

we build a custom byte level BPE tokenizer for all language related models. english only currently. 50176 vocab size.  



#  pretraining (271gb)

aiming for a total 150gb of text for about a currently 2b transformer due to compute capital limitations   
language curriculum (english only for now. agi should be able to learn chinese dynamically during inference, a good test actually) includes stories for a general undertanding of syntax and characters and the world, and then world, which is general formal knowledge about the world like wikipedia or textbooks.       
math curriculum should be by progression of proofs/difficulty. for example algebra should be trained before calculus since calculus requires algebra. there should start with synthetic data perhaps mixed with some word problems then at college level or i guess beyond my level it should just read math arxiv.  
coding will be based on github projects > 50 stars. 3 main languages only. Python, C++, and Javascript. with some html and css for frontend and sql for database. And also bash commands. because python for complex projects with a c++ backend for fast processing / gpu calling is all you need really with additional frontend stuff for a pretty UI. and this is all i use.   
maybe we just mix these in one big pretraining dataset. 


## story (40gb)
TinyStories (500mb stories for kids)  
All The News (10gb news stories from good sources)  
Reddit mix. 17 selected subreddits comments > 5 upvotes. (25gb)
Bookcorpus (5gb self published mid quality books)

## world (75gb)
wikipedia english (19gb complete wikipedia)  
open web text 2 (56gb web articles linked from reddit)  

## math (53 gb)

OpenWebMath (53GB textbooks, stack exchange, latex explanations)  

## coding (105gb)

StarCoderData with repo >= 5 stars of the following languages. all single files:  

StarCoderData Python (73GB)  
StarCoderData C++ (16GB)   
StarCoderData JavaScript (12GB)   
StarCoderData HTML (4.7GB)   
StarCoderData CSS (1.2g)

# SFT (9gb) 
we postrain it with instruction following and conversation 


SlimOrca (3GB chatgpt single turn generated outputs)
OpenAssistant (1GB - human multi-turn quality)
UltraChat (5GB synthetic multi-turn quality)


# Training

we start using RL to train it on specific tasks. like math, code, or others.

## math

synthetic data (5gb code my own with svmbolcore repo. calculation -> algebra -> calculus -> linear algebra progression)