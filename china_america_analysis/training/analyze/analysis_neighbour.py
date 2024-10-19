import gensim
import pandas as pd
from tqdm import tqdm

model_path_template = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/models/word2vec_model_year_{}.model'
years = list(range(1990, 2020))

china_terms = ["china","chinese","beijing","shanghai"]
america_terms = ["america", "usa", "washington", "new york"]
relationship_terms = ["trade", "tariff", "negotiation", "conflict", "collaboration", "cooperation"]


def get_nearest_neighbors(model, term, topn=10):
    try:
        return model.wv.most_similar(term, topn=topn)
    except KeyError:
        return []  

def gather_nearest_neighbors(terms_list, model_path_template, years):
    neighbors_over_time = []
    for year in tqdm(years):
        model_path = model_path_template.format(year)
        try:
            model = gensim.models.Word2Vec.load(model_path)
            neighbors_per_year = {}
            
            for term in terms_list:
                neighbors = get_nearest_neighbors(model, term)
                neighbors_per_year[term] = [neighbor[0] for neighbor in neighbors]
            
            
            neighbors_over_time.append({"Year": year, **neighbors_per_year})
            
        except FileNotFoundError:
            print(f"Model for year {year} not found.")
    return pd.DataFrame(neighbors_over_time)


america_neighbors_df = gather_nearest_neighbors(america_terms, model_path_template, years)
relationship_neighbors_df = gather_nearest_neighbors(relationship_terms, model_path_template, years)
china_neighbors_df = gather_nearest_neighbors(china_terms, model_path_template, years)

china_output_file_path = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/results/china_related_nearest_neighbors_over_time.csv'
america_output_file_path = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/results/america_related_nearest_neighbors_over_time.csv'
relationship_output_file_path = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/results/relationship_related_nearest_neighbors_over_time.csv'

china_neighbors_df.to_csv(china_output_file_path, index=False)
america_neighbors_df.to_csv(america_output_file_path, index=False)
relationship_neighbors_df.to_csv(relationship_output_file_path, index=False)
