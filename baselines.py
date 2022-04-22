
#
# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

import argparse
import spacy
import numpy as np
import pandas as pd
import random
import utils
import os
from collections import Counter
from sklearn.metrics import accuracy_score
from wordfreq import word_frequency

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/preprocessed/train/', help="Directory containing the dataset")
parser.add_argument('--data_dir_stat', default='data/original/english/', help="Directory containing the Wiki dataset")
parser.add_argument('--exercise', default='all')

random.seed(3)

#part B

def process_wiki():
    global wiki_data
    wiki_data.columns = ['target', 'cna', 'cnna', 'bin', 'prob']
    wiki_data['cannotators'] = wiki_data.cna + wiki_data.cnna
    wiki_data['tokens'] = wiki_data.target.apply(lambda x: nlp(x))
    wiki_data['ntokens'] = wiki_data.tokens.apply(lambda x: len(x))

def explore_dataset():
    text = 'Both China and the Philippines flexed their muscles on Wednesday.'
    target = 'flexed their muscles'
    target_pos = text.find(target)
    print('Start and offset for target "' + target + '": ' + str(target_pos) + ' ' + str(target_pos + len(target)))

    target = 'flexed'
    target_pos = text.find(target)
    print('Start and offset for target "' + target + '": ' + str(target_pos) + ' ' + str(target_pos + len(target)))

def basic_stat():
    #7746 in total
    print('Number of instances labeled with 0: %i' % len(wiki_data[wiki_data.bin == 0]))
    print('Number of instances labeled with 1: %i' % len(wiki_data[wiki_data.bin == 1]))
    print('Min, max, median, mean, and stdev of the probabilistic label: %.2f, %.2f, %.2f, %.2f, %.2f' % (
        wiki_data.prob.min(), wiki_data.prob.max(), wiki_data.prob.median(), wiki_data.prob.mean(), wiki_data.prob.std()
    ))
    print('Number of instances consisting of more than one token: %i' % len(wiki_data[wiki_data.ntokens != 1]))
    print('Maximum number of tokens for an instance: %i' % max(wiki_data.ntokens))

def ling_char():
    global wiki_data
    # Filter to take only #tokens = 1 and at least one complex annotation
    wiki_data = wiki_data[(wiki_data.ntokens == 1) & (wiki_data.cannotators > 1)]
    wiki_data['len_tokens'] = wiki_data.tokens.apply(lambda x: len(x[0]))
    wiki_data['freq_tokens'] = wiki_data.tokens.apply(lambda x: word_frequency(str(x[0]), 'en'))
    wiki_data['pos_tag'] = wiki_data.tokens.apply(lambda x: x[-1].pos_)

    print('Pearson correlation length and complexity: ', round(wiki_data.len_tokens.corr(wiki_data.prob),2))
    print('Pearson correlation frequency and complexity: ', round(wiki_data.freq_tokens.corr(wiki_data.prob), 2))

    utils.save_plot(wiki_data.len_tokens, wiki_data.prob, 'length of tokens', 'probabilistic complexity',
                 'Probabilistic complexity by length of tokens', 'len_tokens_prob_scatter.png', 'images/')
    utils.save_plot(wiki_data.freq_tokens, wiki_data.prob, 'frequency of tokens', 'probabilistic complexity',
                 'Probabilistic complexity by frequency of tokens', 'freq_tokens_prob_scatter.png', 'images/')
    utils.save_plot(wiki_data.pos_tag, wiki_data.prob, 'POS tag', 'probabilistic complexity',
                 'Probabilistic complexity by POS tags', 'pos_tags_prob_scatter.png', 'images/')

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def get_labels_in_array(labels):
    y_labels = []
    for row in labels:
        y_labels += row.replace("\n", "").split(" ")
    return y_labels

def get_model_class(train_labels, baseline = 'majority', threshold = 0, token = ''):
    if baseline == 'majority':
        y_train = get_labels_in_array(train_labels)
        model_class = most_frequent(y_train)
    elif baseline == 'random':
        model_class = random.choice(['N', 'C'])
    elif baseline == 'length':
        model_class = 'C' if len(token) > threshold else 'N'
    elif baseline == 'frequency':
        model_class = 'C' if word_frequency(token, 'en') > threshold else 'N'

    return model_class

def baseline(train_labels, test_input, test_labels, baseline = 'majority', threshold = 0):
    model_class = get_model_class(train_labels, 'majority') if baseline == 'majority' else ''
    y_test = get_labels_in_array(test_labels)
    nexp = 100 if baseline == 'random' else 1

    accuracies = []
    # Run experiment
    for i in range(0, nexp):
        # Get predictions
        predictions = []
        words = []
        for instance in test_input:
            tokens = instance.replace('\n','').split(" ")
            instance_predictions = [get_model_class(train_labels, baseline, threshold, t) if baseline != 'majority' else model_class for t in tokens]
            predictions += instance_predictions
            words += tokens
        accuracies.append(accuracy_score(y_test, predictions))

    # Calculate accuracy for the test input
    accuracy = np.mean(accuracies)

    # Build model_output dataframe
    df_out = pd.DataFrame(data=np.array([words, y_test, predictions]).transpose())
    return accuracy, df_out

def get_data():
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
    with open(train_path + "sentences.txt", encoding="utf8", errors='ignore') as sent_file:
        train_sentences = sent_file.readlines()
    with open(train_path + "labels.txt", encoding="utf8", errors='ignore') as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt", encoding="utf8", errors='ignore') as dev_file:
        dev_sentences = dev_file.readlines()
    with open(dev_path + "labels.txt", encoding="utf8", errors='ignore') as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt", encoding="utf8", errors='ignore') as test_file:
        test_sentences = test_file.readlines()
    with open(test_path + "labels.txt", encoding="utf8", errors='ignore') as test_label_file:
        test_labels = test_label_file.readlines()

    return (train_sentences, train_labels, dev_sentences, dev_labels, test_sentences, test_labels)

def run_threshold_exp(train_labels, dev_sentences, dev_labels):
    experiments = {'length': range(1, 21), 'frequency': np.arange(0, 0.1, 0.005)}

    for bsln, thresholds in experiments.items():
        print('Run threshold experiment for %s ......' % bsln)
        for thrsh in thresholds:
            accuracy, df_out = baseline(train_labels, dev_sentences, dev_labels, bsln, thrsh)
            print('Accuracy on dev, %s threshold %.3f baseline: %.4f' % (bsln, thrsh, accuracy))
        print("\n")

def written_ex():
    print('Exercise 9 is written in the report')

def create_baselines():
    test_out_path = "experiments/"

    train_sentences, train_labels, dev_sentences, dev_labels, test_sentences, test_labels = get_data()
    run_threshold_exp(train_labels, dev_sentences, dev_labels)

    # Run all baselines
    datasets = {'dev': {'sentences': dev_sentences, 'labels': dev_labels},
                'test': {'sentences': test_sentences, 'labels': test_labels}}
    thresholds = {'majority': 0, 'random': 0, 'length': 8, 'frequency': 0.055}

    for env in ['dev', 'test']:
        for bsln in ['majority', 'random', 'length', 'frequency']:
            accuracy, df_out = baseline(train_labels, datasets[env]['sentences'], datasets[env]['labels'], bsln,
                                        thresholds[bsln])
            print('Accuracy on %s, %s baseline: %.2f' % (env, bsln, accuracy))
            if env == 'test':
                if not os.path.exists(test_out_path + bsln + '_model'):
                    os.makedirs(test_out_path + bsln + '_model')
                df_out.to_csv(test_out_path + bsln + "_model/model_output.tsv", sep="\t", index=False, header=False)
        print('\n')

if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()

    if args.exercise == 'all':
        exercise = range(6, 11)
    else:
        exercise = [int(args.exercise)]

    # Load and Process Wiki Data
    if any(ex in [7, 8] for ex in exercise):
        print('Processing wiki file...')
        nlp = spacy.load("en_core_web_sm")
        wiki_data = pd.read_csv(args.data_dir_stat + "WikiNews_Train.tsv", sep='\t', header=None,
                                usecols=[4, 7, 8, 9, 10])
        process_wiki()

    # Load function names for each exercise
    functions = {6: explore_dataset, 7: basic_stat, 8: ling_char, 9: written_ex, 10: create_baselines}

    for ex in exercise:
        print('-----------------Running exercise %i-----------------' % ex)
        functions[ex]()
        print('\n')