import nltk
nltk.download('punkt_tab')
nltk.download('popular')


from nltk.tokenize import sent_tokenize

# convert text to sentences
f = open("demotext.txt", "r")
text = f.read()

sentences = sent_tokenize(text)


from functions import *

# 1 text preprocessing 
text_preprocessing(sentences)

# 2 word embeddings 
word_embeddings = extract_word_vectors()

# 3 sentence vector embeddings 
sentence_vectors = sentence_vector_representation(sentences,word_embeddings)

# 4 create similarity matrix 
sim_mat = create_similarity_matrix(sentences, sentence_vectors)

# 5 determine sentence rank 
ranked_sentences = determine_sentence_rank(sentences, sim_mat)

# 6 generate summary
summary = generate_summary(sentences, ranked_sentences)

f = open("result_demotext.txt", "w")
f.write(summary)
f.close()
