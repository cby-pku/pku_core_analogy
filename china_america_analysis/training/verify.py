from gensim.models import Word2Vec


model_path = '/data/align-anything/boyuan/core_workspace/china_america_analysis/training/models/word2vec_model_year_2015.model'
model = Word2Vec.load(model_path)
print("Model loaded successfully!")

def log(string):
    print()
    print('**'*20)
    print(string)
    print('**'*20)
    print()

# NOTE 1 Check Similar Words

log('Task1: Checking Similar Words...')
word = 'technology'
if word in model.wv:
    similar_words = model.wv.most_similar(word, topn=5)
    print(f"Most similar words to '{word}':")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity:.4f}")
else:
    print(f"The word '{word}' is not in the vocabulary.")

# NOTE 2 Compute Similarity
log('Task2: Computing Similarity...')
word1 = 'cat'
word2 = 'dog'
if word1 in model.wv and word2 in model.wv:
    similarity = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
else:
    print(f"One of the words '{word1}' or '{word2}' is not in the vocabulary.")

# NOTE 3 Analogy Reasoning
log('Task3: Analogy Reasoning...')
positive = ['paris', 'france']
negative = ['berlin']
if all(word in model.wv for word in positive + negative):
    result = model.wv.most_similar(positive=positive, negative=negative, topn=1)
    print(f"'{positive[0]}' is to '{positive[1]}' as '{negative[0]}' is to '{result[0][0]}'")
else:
    print("One of the words is not in the vocabulary.")

# NOTE 4 Check Word Vectors
log('Task4: Check Word Vectors...')
word = 'science'
if word in model.wv:
    vector = model.wv[word]
    print(f"Vector representation of '{word}':")
    print(vector)
else:
    print(f"The word '{word}' is not in the vocabulary.")
