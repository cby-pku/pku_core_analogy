##  Word2Vec Training for English Analysis



### Basic Usage
1. `bash train.sh` for training the word2vec model.
2. After getting the model, you can first use `verify.py` to easily check the basic performance of the model
3. Using `analyze/change_analysis.py` to analyze china_terms \ america_terms \ relationship_terms changing tendency (log: at 1019 22:39, but I think maybe there is a more effecitve way.)

### Analysis

1. `analyze/change_tendency.py` to analyze the alignment & divergence and sentiment proximity



"""
Explanation:
1. **Alignment and Divergence Over Time**: This plot shows the cosine distance between key terms over time. 
   - The distance between "China" and "cooperation" represents how aligned these two terms are in the word embedding space.
   - The distance between "America" and "conflict" represents how closely associated these terms are.
   - A lower cosine distance indicates a stronger alignment or connection between the terms, while a higher distance indicates divergence.

2. **Sentiment Proximity Over Time**: This plot shows how sentiment-related words are associated with "China" and "America."
   - "China Sentiment Proximity" is the average cosine similarity between "China" and sentiment words like "positive," "negative," "war," and "peace."
   - "America Sentiment Proximity" represents the same for "America."
   - Higher sentiment proximity indicates a stronger association between these countries and the sentiment-related terms.
"""