# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

import numpy as np
import random
from collections import Counter
from sklearn.metrics import accuracy_score

from model.data_loader import DataLoader

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

def majority_baseline(train_sentences, train_labels, test_input, test_labels):
    y_test = get_labels_in_array(test_labels)

    # Determine the majority class based on the training data
    y_train = get_labels_in_array(train_labels)
    majority_class = most_frequent(y_train)

    # Get predictions
    predictions = []
    for instance in test_input:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions += instance_predictions

    # Calculate accuracy for the test input
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

def random_baseline(train_sentences, train_labels, test_input, test_labels):
    y_test = get_labels_in_array(test_labels)

    accuracies = []
    for i in range(0,100):
        # Get predictions
        predictions = []
        for instance in test_input:
            tokens = instance.split(" ")
            instance_predictions = [random.choice(['N', 'C']) for t in tokens]
            predictions += instance_predictions
        accuracies.append(accuracy_score(y_test, predictions))

    # Calculate accuracy for the test input
    accuracy = np.mean(accuracies)

    return accuracy, predictions

if __name__ == '__main__':
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
        test_input = test_file.readlines()
    with open(test_path + "labels.txt", encoding="utf8", errors='ignore') as test_label_file:
        test_labels = test_label_file.readlines()

    # Majority Baseline, Dev and Test:
    majority_accuracy, majority_predictions = majority_baseline(dev_sentences, dev_labels, test_input, test_labels)
    print('Accuracy on dev, majority baseline: %.2f' % majority_accuracy)
    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, test_input, test_labels)
    print('Accuracy on test, majority baseline: %.2f' % majority_accuracy)

    # Random Baseline, Dev and Test:
    random_accuracy, random_predictions = random_baseline(dev_sentences, dev_labels, test_input, test_labels)
    print('Accuracy on dev, random baseline: %.2f' % random_accuracy)
    random_accuracy, random_predictions = random_baseline(train_sentences, train_labels, test_input, test_labels)
    print('Accuracy on test, random baseline: %.2f' % random_accuracy)

    # TODO: output the predictions in a suitable way so that you can evaluate them