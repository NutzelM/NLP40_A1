from decimal import DivisionByZero
from importlib.resources import open_text
from shutil import which
import spacy 
from collections import Counter
import numpy as np
import pandas as pd
from nltk import ngrams
from nltk import FreqDist

training_data_file = open("data/preprocessed/train/sentences.txt", "r")
training_data  = training_data_file.read()
training_data_file.close()
nlp = spacy.load('en_core_web_sm')
doc = nlp(training_data)

which_exercise = [5] # you can change this to only run one or less for efficiency :)

# 1 -----------------------
if 1 in which_exercise:
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


if 2 in which_exercise:
    ## 2 -----------------------
    # DO THEY ONLY WANT WORDS OR ALSO PUNCTIATION
    fPOS_frequencies = Counter()
    fPOS = []
    # dictionary for fine --> unif.
    unfPOS_dict = {}
    for sentence in doc.sents:
        for token in sentence:
            fPOS.append(token.tag_)
            if token.tag_ not in unfPOS_dict.keys():
                unfPOS_dict[token.tag_] = token.pos_

    fPOS_frequencies.update(fPOS)
    #print(fPOS_frequencies)

    fPOS10 = []
    occ10 = []
    tag_freq = []
    for fPOStag, cnt in fPOS_frequencies.most_common(10):
        fPOS10.append(fPOStag) #Finegrained POS tag
        occ10.append(cnt) # occurancy of tag
        tag_freq.append((cnt/num_tokens)) # frequency of tag

    def findWordsTag(doc, tag): 
        word_frequencies_tag = Counter()
        for sentence in doc.sents:
            words_tag= []
            for token in sentence: 
                if token.tag_ == tag:
                    words_tag.append(token.text)
            word_frequencies_tag.update(words_tag)
        return word_frequencies_tag

    # find 3 most and least frequent words with tags in fPOS10
    most_freq_tokens = []
    least_freq_tokens = []
    unfPOS10 = []
    for tag in fPOS10:
        unfPOS10.append(unfPOS_dict[tag])
        freq_token_tag = findWordsTag(doc, tag)
        freq_tokens = []
        print(f"for tag {tag} the most common are : \n {freq_token_tag.most_common(3)} ")
        for token, cnt in freq_token_tag.most_common(3):
            freq_tokens.append(token)
        most_freq_tokens.append((tuple(freq_tokens)))
        least_freq_tokens.append(freq_token_tag.most_common()[:-2:-1][0][0])

    most_freq_words_tag =  findWordsTag(doc, fPOS10[0])
    class_table = pd.DataFrame({'Finegrained POS-tag' : fPOS10 ,'Universal POS-tag' : unfPOS10, 'Occurances': occ10, 'Relative Tag Frequency (%)' : tag_freq, '3 most frequent tokens' : most_freq_tokens, 'Infrequent token': least_freq_tokens} )
    print(class_table)
#3 ------------------------
if 3 in which_exercise:
   
    def get_ngram(text, ngram):
        temp = zip(*[text[i:] for i in range(0, ngram)])
        return [' '.join(ngram) for ngram in temp]
        
    pos = []
    tokens = []
    for token in doc:
        if not token.is_space:
            pos.append(token.tag_)
            tokens.append(token.text)

    bigram_tokens = Counter(get_ngram(tokens, 2))
    trigram_tokens = Counter(get_ngram(tokens, 3))

    bigram_pos = Counter(get_ngram(pos, 2))
    trigram_pos = Counter(get_ngram(pos, 3))

    #TO-DO: Maike, Giulia - speak about Maike taking unigrams for bigrams and bigrams for trigrams
    print('Token bigrams: ', bigram_tokens.most_common(3))
    print('Token trigrams: ', trigram_tokens.most_common(3))
    print('POS bigrams: ', bigram_pos.most_common(3))
    print('POS trigrams:', trigram_pos.most_common(3))
    print('\n')

# 4.	Lemmatization (1 point)
# Provide an example for a lemma that occurs in more than two inflections in the dataset. 
# Lemma:
# Inflected Forms: 
# Example sentences for each form: 



#-----------
if 4 in which_exercise:
    def find3inflictions(doc):
        tokens = {}
        sentences = {}
        for sentence in doc.sents:
            for token in sentence:
                if (token.lemma_ != token.text): #then there is an infiction
                    if token.lemma_ not in tokens.keys():
                        tokens[token.lemma_] = [token.text]
                        sentences[token.lemma_] = [sentence]
                    else:
                        # if infliction did not exist, add to list
                        if token.text not in tokens[token.lemma_]:
                            tokens[token.lemma_].append(token.text)
                            sentences[token.lemma_].append(sentence)
                    if len(tokens[token.lemma_]) == 3:
                        return token.lemma_, tokens[token.lemma_], sentences[token.lemma_]
        return None

    lemma, words, sentences = find3inflictions(doc)  
    print(lemma)
    print(words)
    print(f"Lemma: {lemma},\n Inflected Forms: {words}, \n Example sentences for each form: \n {sentences}")

#--- 
if 5 in which_exercise:
    freq_ents = Counter()
    freq_labels = Counter()
    for sentence in doc.sents:
        for ent in sentence.ents:
            if ent.text != '/n' and ent.text != '\\':
                freq_ents.update([ent.text])
               # print(ent.text)
                freq_labels.update([ent.label_])
        num_ents = len(freq_ents)
        num_labels = len(freq_labels)
    print(f"Number of named entities: {num_ents}, \n Number of different entity labels:  {num_labels}")

