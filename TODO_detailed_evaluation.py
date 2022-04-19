import pandas as pd 
import numpy as np


def evaluate_predictions_detailed(df, pos, neg):
    df.columns = ['word', 'gold', 'prediction']
    cm = confusion_matrix(df, pos, neg)
    pr = precision(cm)
    rec = recall(cm)
    f1mes = f1(pr, rec)
    return pr, rec, f1mes

def precision(cm):
    return cm[0,0]/(cm[0,0] + cm[1,0])

def recall(cm):
    return cm[0,0]/(cm[0,0] + cm[0,1])

def f1(pr, rec):
    return (pr * rec)/(pr + rec)

def confusion_matrix(df, pos, neg):
    P = np.sum(df.prediction == pos)
    TP = np.sum ((df.prediction == pos) & (df.gold == pos))
    FP = np.sum((df.prediction == pos) & (df.gold == neg))
    # if not the case something went wrong
    assert (TP + FP) == P
    
    N = np.sum(df.prediction == neg)
    TN = np.sum ((df.prediction == neg) & (df.gold == neg))
    FN = np.sum((df.prediction == neg) & (df.gold == pos))
    # if not the case something went wrong
    assert (TN + FN) == N
    return np.matrix([[TP, FN], [FP, TN]])

def find_metric(df):
    # for Class N
    pr_N, rec_N, f1mes_N = evaluate_predictions_detailed(df, 'N', 'C')
    # for Class C
    pr_C, rec_C, f1mes_C = evaluate_predictions_detailed(df, 'C', 'N')
    
    return [pr_N, rec_N, f1mes_N, pr_C, rec_C, f1mes_C]

if __name__ == '__main__':
    """
        Gives detailed evaluation on predictions from evaluate.py
    """
    df = pd.read_csv("experiments/base_model/model_output.tsv", sep='\t')
    print(find_metric(df))


    
