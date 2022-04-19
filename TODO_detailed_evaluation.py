import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report

def evaluate_predictions_detailed(df, pos, neg):
    df.columns = ['word', 'gold', 'prediction']
    cm = confusion_matrix(df, pos, neg)
    pr = precision(cm)
    rec = recall(cm)
    f1mes = f1(pr, rec)
    return pr, rec, f1mes

def precision(cm):
    return round(cm[0,0]/(cm[0,0] + cm[1,0]),2)

def recall(cm):
    return round(cm[0,0]/(cm[0,0] + cm[0,1]),2)

def f1(pr, rec):
    return round(2*(pr * rec)/(pr + rec),2)

def confusion_matrix(df, pos, neg):
    P = np.sum(df.prediction == pos)
    TP = np.sum ((df.prediction == pos) & (df.gold == pos))
    FP = np.sum((df.prediction == pos) & (df.gold == neg))
    # if not the case something went wrong
    assert (TP + FP) == P
    
    N = np.sum(df.prediction == neg)
    TN = np.sum((df.prediction == neg) & (df.gold == neg))
    FN = np.sum((df.prediction == neg) & (df.gold == pos))
    # if not the case something went wrong
    assert (TN + FN) == N
    return np.matrix([[TP, FN], [FP, TN]])

def find_metric(df):
    # for Class N
    pr_N, rec_N, f1mes_N = evaluate_predictions_detailed(df, 'N', 'C')
    # for Class C
    pr_C, rec_C, f1mes_C = evaluate_predictions_detailed(df, 'C', 'N')
    # Weighted Average F1
    nn = np.sum(df.prediction == 'N')
    nc = np.sum(df.prediction == 'C')
    wa_f1 = round((f1mes_N * nn + f1mes_C * nc) / (nn + nc),2)
    
    return [pr_N, rec_N, f1mes_N, pr_C, rec_C, f1mes_C, wa_f1]

if __name__ == '__main__':
    """
        Gives detailed evaluation on predictions from evaluate.py
    """
    df = pd.read_csv("experiments/base_model/model_output.tsv", sep='\t')
    print(find_metric(df))

    # Double check the values obtained
    df.dropna(subset=["prediction"], inplace=True)
    print(classification_report(np.array(df.gold), np.array(df.prediction), digits=2))
    
