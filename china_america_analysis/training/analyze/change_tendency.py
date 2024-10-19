import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Define the path template for the Word2Vec models and the years range
model_path_template = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/models/word2vec_model_year_{}.model'
years = list(range(1990, 2020))


def cosine_distance(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_alignment_divergence(term_pairs, model_path_template, years):
    alignment_data = []
    
    for year in tqdm(years):
        model_path = model_path_template.format(year)
        try:
            model = gensim.models.Word2Vec.load(model_path)
            yearly_data = {"Year": year}
            
            for term1, term2 in term_pairs:
                try:
                    vec1 = model.wv[term1]
                    vec2 = model.wv[term2]
                    distance = cosine_distance(vec1, vec2)
                    yearly_data[f"{term1}-{term2}_distance"] = distance
                except KeyError:
                    yearly_data[f"{term1}-{term2}_distance"] = None
            
            alignment_data.append(yearly_data)
            
        except FileNotFoundError:
            print(f"Model for year {year} not found.")
    
    return pd.DataFrame(alignment_data)

# Function to calculate sentiment proximity over time
def calculate_sentiment_proximity(key_terms, sentiment_terms, model_path_template, years):
    sentiment_data = []
    
    for year in tqdm(years):
        model_path = model_path_template.format(year)
        try:
            model = gensim.models.Word2Vec.load(model_path)
            yearly_data = {"Year": year}
            
            for key_term in key_terms:
                similarities = []
                
                for sentiment_term in sentiment_terms:
                    try:
                        vec1 = model.wv[key_term]
                        vec2 = model.wv[sentiment_term]
                        similarity = 1 - cosine_distance(vec1, vec2)
                        similarities.append(similarity)
                    except KeyError:
                        continue
                
                if similarities:
                    yearly_data[f"{key_term}_sentiment_proximity"] = np.mean(similarities)
                else:
                    yearly_data[f"{key_term}_sentiment_proximity"] = None
            
            sentiment_data.append(yearly_data)
            
        except FileNotFoundError:
            print(f"Model for year {year} not found.")
    
    return pd.DataFrame(sentiment_data)


# Calculate alignment and divergence for key China and America terms # 词向量的聚类和发散
# NOTE change this one, and check 变化
term_pairs = [("china", "cooperation"), ("china", "conflict")]
alignment_df = calculate_alignment_divergence(term_pairs, model_path_template, years)

# Calculate sentiment proximity for China and America
key_terms = ["china", "america"]
sentiment_terms = ["positive", "peace"]
sentiment_proximity_df = calculate_sentiment_proximity(key_terms, sentiment_terms, model_path_template, years)

alignment_output_file_path = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/results/alignment_divergence_over_time_1.csv'
sentiment_output_file_path = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/results/sentiment_proximity_over_time_1.csv'

alignment_df.to_csv(alignment_output_file_path, index=False)
sentiment_proximity_df.to_csv(sentiment_output_file_path, index=False)



# Ploting the results

# Plot the alignment/divergence curve
plt.figure(figsize=(10, 6))
plt.plot(alignment_df['Year'], alignment_df['china-cooperation_distance'], label='China-Cooperation Distance', marker='o')
plt.plot(alignment_df['Year'], alignment_df['china-conflict_distance'], label='China-Conflict Distance', marker='o')
plt.title('Alignment and Divergence Over Time')
plt.xlabel('Year')
plt.ylabel('Cosine Distance')
plt.legend()
plt.grid(True)
output_folder = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/figures'
os.makedirs(output_folder,exist_ok = True)
plt.savefig(os.path.join(output_folder,'china_alignment_divergence_score.png'),dpi=500)
plt.show()

# Plot the sentiment proximity curve
plt.figure(figsize=(10, 6))
plt.plot(sentiment_proximity_df['Year'], sentiment_proximity_df['china_sentiment_proximity'], label='China Sentiment Proximity', marker='o')
plt.plot(sentiment_proximity_df['Year'], sentiment_proximity_df['america_sentiment_proximity'], label='America Sentiment Proximity', marker='o')
plt.title('Sentiment Proximity Over Time (china-america-positive-peace)')
plt.xlabel('Year')
plt.ylabel('Sentiment Proximity (Cosine Similarity)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder,'china_america_postive_peace_sentiment_proximity_curve.png'),dpi=500)
plt.show()



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
