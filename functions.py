from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


wl = WordNetLemmatizer()
def text_preprocessing(sentences: list) -> list:
    """
    Pre processing text to remove unnecessary words.
    """
    print('Preprocessing text')

    stop_words = set(stopwords.words('english'))

    clean_words = None
    for sent in sentences:
        words = word_tokenize(sent)
        words = [wl.lemmatize(word.lower()) for word in words if word.isalnum()]
        clean_words = [word for word in words if word not in stop_words]

    return clean_words

import numpy as np
def extract_word_vectors() -> dict:
    """
    Extracting word embeddings. These are the n vector representation of words.
    """
    print('Extracting word vectors')

    word_embeddings = {}
    # Here we use glove word embeddings of 100 dimension
    f = open('word vector embeddings/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs

    f.close()
    return word_embeddings

def sentence_vector_representation(sentences: list, word_embeddings: dict) -> list:
    """
    Creating sentence vectors from word embeddings.
    """
    print('Sentence embedding vector representations')

    sentence_vectors = []
    for sent in sentences:
        clean_words = text_preprocessing([sent])
        # Averaging the sum of word embeddings of the sentence to get sentence embedding vector
        v = sum([word_embeddings.get(word, np.zeros(100, )) for word in clean_words]) / (len(clean_words) + 0.001)
        sentence_vectors.append(v)

    return sentence_vectors
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity_matrix(sentences: list, sentence_vectors: list) -> np.ndarray:
    """
    Using cosine similarity, generate similarity matrix.
    """
    print('Creating similarity matrix')

    # Defining a zero matrix of dimension n * n
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                # Replacing array value with similarity value.
                # Not replacing the diagonal values because it represents similarity with its own sentence.
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    return sim_mat

import networkx as nx
def determine_sentence_rank(sentences: list, sim_mat: np.ndarray) -> list:
    """
    Determining sentence rank using Page Rank algorithm.
    """    
    print('Determining sentence ranks')
    
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted([(scores[i], s[:15]) for i, s in enumerate(sentences)], reverse=True)
    return ranked_sentences


def generate_summary(sentences: list, ranked_sentences: list) -> str:
    """
    Generate a sentence for sentence score greater than average.
    """
    print('Generating summary')

    # Get top ranked sentences (1/3 th) to generate 1/3 of the original text.
    top_ranked_sentences = ranked_sentences[:int(len(sentences)/ 3)]

    sentence_count = 0
    summary = ''

    for i in sentences:
        for j in top_ranked_sentences:
            if i[:15] == j[1]:
                summary += i + ' '
                sentence_count += 1
                break

    return summary