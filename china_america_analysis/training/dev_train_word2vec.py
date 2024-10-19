import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
import numpy as np
import argparse
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os

def preprocess_text(text):
    text = re.sub(r'@@[0-9]+', '', text)  # Remove meaningless symbols
    text = re.sub(r'<[^>]+>', '', text)    # Remove HTML tags
    return simple_preprocess(text)

import gensim

def evaluate_model(model):
    from gensim.test.utils import datapath
    ws353 = datapath('wordsim353.tsv')
    analogy_path = datapath('questions-words.txt')

    # Word similarity evaluation
    sim_corr = model.wv.evaluate_word_pairs(ws353)
    pearson_corr = sim_corr[0][0]
    spearman_corr = sim_corr[1][0]
    oov_ratio = sim_corr[2]

    # Word analogy evaluation
    gensim_version = gensim.__version__
    major_version = int(gensim_version.split('.')[0])

    if major_version >= 4:
        analogy_score, analogy_result = model.wv.evaluate_word_analogies(analogy_path)
        # analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
       #  print((analogy_score))
        analogy_accuracy = analogy_score
    else:
        analogy_result = model.wv.evaluate_word_analogies(analogy_path)
        correct = 0
        total = 0
        for section in analogy_result[1]:
            correct += len(section['correct'])
            total += len(section['correct']) + len(section['incorrect'])
        analogy_accuracy = correct / total if total > 0 else 0

    return pearson_corr, spearman_corr, oov_ratio, analogy_accuracy



def train_word2vec(train_sentences, val_sentences, config,year):
    model = Word2Vec(
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=config['min_count'],
        workers=4,
        compute_loss=True
    )

    model.build_vocab(train_sentences)

    cumulative_loss = 0  

    for epoch in tqdm(range(config['epochs'])):
        print(f"Training epoch {epoch + 1}/{config['epochs']}")
        model.train(
            train_sentences,
            total_examples=model.corpus_count,
            epochs=1,
            compute_loss=True
        )

        current_loss = model.get_latest_training_loss()
        epoch_loss = current_loss - cumulative_loss
        cumulative_loss = current_loss  

        print(f"Epoch {epoch + 1} Loss: {epoch_loss}")

        pearson_corr, spearman_corr, oov_ratio, analogy_accuracy = evaluate_model(model)
        wandb.log({
            "train_loss": epoch_loss,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "oov_ratio": oov_ratio,
            "analogy_accuracy": analogy_accuracy,
            "epoch": epoch + 1
        })

    model.save(f"./models/word2vec_model_year_{year}.model")
    print("Model training complete.")

def load_data(data_path):

    with open(data_path, 'r') as f:
        raw_text = f.read()
    sentences = sent_tokenize(raw_text)
    processed_sentences = [preprocess_text(sentence) for sentence in sentences if sentence.strip()]
    return processed_sentences

def main(args):
    
    wandb.init(project="word2vec-experiment", config={
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs
    }, name=args.run_name)

    sentences = load_data(args.input_file)
    train_sentences, val_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
    train_word2vec(train_sentences, val_sentences, wandb.config,args.year)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Word2Vec model with wandb logging")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--year", type=str, required=True, help="Year of the training dataset")
    parser.add_argument("--vector_size", type=int, default=100, help="Word vector size (default: 100)")
    parser.add_argument("--window", type=int, default=5, help="Window size for Word2Vec (default: 5)")
    parser.add_argument("--min_count", type=int, default=10, help="Minimum word frequency (default: 5)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train (default: 20)")
    parser.add_argument("--run_name", type=str, default="word2vec_run", help="Name of the wandb run")

    args = parser.parse_args()
    main(args)
