from importlib.resources import open_text
import spacy 
from collections import Counter
import numpy as np

training_data_file = open("data/preprocessed/train/sentences.txt", "r")
training_data  = training_data_file.read()
training_data_file.close()
nlp = spacy.load('en_core_web_sm')
doc = nlp(training_data)

## 1 -----------------------

# computes word frequences, outputs dict.
def numFrequencies(doc):
    word_frequencies = Counter()
    for sentence in doc.sents:
        words = []
        for token in sentence: 
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
        word_frequencies.update(words)
    return word_frequencies

# Number of tokens 
def numTokens(doc):
    return len(doc)

def numWordsAndTypes(doc):
    word_frequencies = numFrequencies(doc)
    return sum(word_frequencies.values()) , len(word_frequencies.keys())


# Number of types : all the words without punctuation and repeat
# Number of words : all the words without punctuation etc 
num_tokens = numTokens(doc)
num_words, num_types = numWordsAndTypes(doc)


# Average number of words per sentence
sentences = doc.sents
num_words_sentence_all = []
for sentence in sentences:
    num_words_sentence, num_types_sentence = numWordsAndTypes(sentence)
    num_words_sentence_all.append(num_words_sentence)

mean_num_words_per_sentence = np.mean(num_words_sentence_all)

# Average word length
all_words = numFrequencies(doc).keys()
word_length = []
for word in all_words:
    word_length.append(len(word))

mean_word_lenth = np.mean(word_length)


print(f"Number of tokens: \n {num_tokens} \n Number of types: \n {num_types} \n Number of words: \n {num_words} \n Average number of words per sentence: \n {mean_num_words_per_sentence} \n Average word length: {mean_word_lenth}\n")



