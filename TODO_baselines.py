# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

import numpy as np
import pandas as pd
import random
import os
from collections import Counter
from sklearn.metrics import accuracy_score
from wordfreq import word_frequency

random.seed(3)

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
            tokens = instance.split(" ")
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
    global train_sentences, train_labels, dev_sentences, dev_labels, test_sentences, test_labels
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

def run_threshold_exp():
    experiments = {'length': range(1, 21), 'frequency': np.arange(0, 0.1, 0.005)}

    for bsln, thresholds in experiments.items():
        print('Run threshold experiment for %s ......' % bsln)
        for thrsh in thresholds:
            accuracy, df_out = baseline(train_labels, dev_sentences, dev_labels, bsln, thrsh)
            print('Accuracy on dev, %s threshold %.3f baseline: %.4f' % (bsln, thrsh, accuracy))
        print("\n")

if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"
    test_out_path = "experiments/"

    get_data()
    run_threshold_exp()

    # Run all baselines
    datasets = {'dev': {'sentences': dev_sentences, 'labels': dev_labels},
                'test': {'sentences': test_sentences, 'labels': test_labels}}
    thresholds = {'majority': 0, 'random': 0, 'length': 8, 'frequency': 0.055}

    for env in ['dev', 'test']:
        for bsln in ['majority', 'random', 'length', 'frequency']:
            accuracy, df_out = baseline(train_labels, datasets[env]['sentences'], datasets[env]['labels'], bsln, thresholds[bsln])
            print('Accuracy on %s, %s baseline: %.2f' % (env, bsln, accuracy))
            if env == 'test':
                if not os.path.exists(test_out_path + bsln + '_model'):
                    os.makedirs(test_out_path + bsln + '_model')
                df_out.to_csv(test_out_path + bsln + "_model/model_output.tsv", sep="\t", index=False, header=False)
        print('\n')