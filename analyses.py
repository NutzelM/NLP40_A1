import argparse
import spacy
import numpy as np
import pandas as pd
from collections import Counter

# Pandas settings
pd.set_option('display.max_columns', None)

# Run with arguments, for example: --exercise 3
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/preprocessed/train/', help="Directory containing the dataset")
parser.add_argument('--data_dir_stat', default='data/original/english/', help="Directory containing the Wiki dataset")
parser.add_argument('--exercise', default='all')

def get_words_data(text):
    words, lens = [], []
    for token in text:
        if token.is_punct != True:
            words.append(token.text)
            lens.append(len(token.text))
    if len(lens) != 0:
        avg_len = sum(lens) / len(lens)
    else:
        avg_len = 0
    return len(words), avg_len

def get_average_num_words(text):
    sentences = list(text.sents)
    avg = []
    for sent in sentences:
        avg.append(get_words_data(sent)[0])
    return sum(avg) / len(avg)

def tokenization():
    # The number of tokens such as words, numbers, punctuation marks etc.
    tokens = [token.text for token in doc]
    print("Number of tokens: %i" % len(tokens))
    # The number of unique tokes
    print("Number of types: %i" % len(set(tokens)))
    words_data = get_words_data(doc)
    # The number of words after removing stopwords and punctuations
    print("Number of words: %i" % words_data[0])
    # The number of average words per sentence
    print("Average number of words per sentence: %.2f" % get_average_num_words(doc))
    print("Average word length: %.2f" % words_data[1])

def words():
    tags = []
    for token in doc:
        if not token.is_space:
            tags.append(token.tag_)

    # Get 10 most frequent tags
    unique_elements, frequency = np.unique(tags, return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    fgPOS = unique_elements[sorted_indexes][:10]
    freq = frequency[sorted_indexes][:10]

    freq_tokens, infreq_tokens, uPOS = [], [], []
    for tag in fgPOS:
        # Get most frequent and infrequent words with that tag
        words = [token.text for token in doc if token.tag_ == tag]
        words_tally = Counter(words)
        freq_tokens.append(', '.join([word for word, cnt in words_tally.most_common(3)]))
        infreq_tokens.append(words_tally.most_common()[-1][0])
        # Get POS for that tag
        uPOS.append(next(token.pos_ for token in doc if token.tag_ == tag))

    # Build DataFrame for output
    word_class = pd.DataFrame({'Fg POS-tag': fgPOS,
                               'Universal POS-tag': uPOS,
                               'Occurrences': freq[:10],
                               'Relative Tag Freq(%)': np.around(freq[:10] / len(tags),2),
                               '3 most frequent tokens': freq_tokens,
                               'Example infrequent token': infreq_tokens})
    print(word_class)

def get_ngram(text, ngram):
    temp = zip(*[text[i:] for i in range(0, ngram)])
    return [' '.join(ngram) for ngram in temp]

def ngrams():
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

def lemmatization():
    tokens = {}
    sentences = {}
    for sentence in doc.sents:
        for token in sentence:
            if (not token.is_space and (token.lemma_ != token.text.lower())):  # then there is an inflection
                if token.lemma_ not in tokens.keys():
                    tokens[token.lemma_] = [token.text]
                    sentences[token.lemma_] = [sentence]
                else:
                    # if infliction did not exist, add to list
                    if token.text not in tokens[token.lemma_]:
                        tokens[token.lemma_].append(token.text)
                        sentences[token.lemma_].append(sentence)
                if len(tokens[token.lemma_]) == 3:
                    print('Lemma: ', token.lemma_)
                    print('Inflected Forms: ', tokens[token.lemma_])
                    print('Example sentences for each form: ', sentences[token.lemma_])
                    return

def ner():
    ne, ne_labels = [], []
    for ent in doc.ents:
        ne.append(ent.text)
        ne_labels.append(ent.label_)
    print('Number of named entities: ', len(ne))
    print('Number of unique named entities: ', len(set(ne)))
    print('Number of different entity labels: ', len(set(ne_labels)))

    for i, sentence in enumerate(doc.sents):
        print(sentence)
        print('Named entities: ', [ent.text for ent in sentence.ents])
        if i==4: break

if __name__ == '__main__':
    """
        `Data Analysis`
    """
    # Load the parameters
    args = parser.parse_args()
    args_dict = vars(args)

    if args.exercise == 'all':
        exercise = range(1, 6)
    else:
        exercise = [int(args.exercise)]

    # Load the data
    data_file = open(args.data_dir + "sentences.txt", encoding="utf8", errors='ignore')
    data = data_file.read()
    data_file.close()

    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(data)

    # Load function names for each exercise
    functions = {1: tokenization, 2: words, 3: ngrams, 4: lemmatization, 5: ner}

    for ex in exercise:
        print('-----------------Running exercise %i-----------------' % ex)
        functions[ex]()
        print('\n')



