This project contains code for part 1 of the course Natural Language Processing Technology 2022 at VU Amsterdam put together by Lisa Beinborn.
This particular file contains the code and answers for assignment 1 for the NLP course 2022, written by by Andreea Hazu, Maike Nützel & Giulia Bössenecker

Important:
default_analyses.py can be ignored.
The code for exercise 14 can be found in experiments.py

References:
The modelling part draws substantially on the code project for the Stanford course cs230-stanford developed by Surag Nair, Guillaume Genthial and Olivier Moindrot (https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp).
It has been simplified and adapted by Lisa Beinborn.

The data in data/original is a small subset from http://sites.google.com/view/cwisharedtask2018/. Check data/original/README.md for details.

The data in data/preprocessed has been processed by Sian Gooding for her submission to the shared task (https://github.com/siangooding/cwi/tree/master/CWI%20Sequence%20Labeller).

Task:
Make sure to install the libraries in requirements.txt.

- For part A of the assignment, linguistic analyses was provided in in analyses.py using spacy.
- For part B of the assignment, we calculated baselines in baselines.py.
- For part C of the assignment, we built the vocabulary and trained the model.
- We inplemented functions to evaluate the output of the LTSM model and the baselines in detailed_evaluation.py.
- Additional experiments were ran for hyperparameter tunning in experiments.py.
