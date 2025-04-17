import os
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
import html
import emoji
import math
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

class LR:
    def __init__(self, TRAIN_SIZE):
        self.model      = LogisticRegression(solver='saga', max_iter=2000)
        self.vocab      = None
        self.token2idx  = None

    def set_vocab(self, vocab_list):
        self.vocab     = list(vocab_list)
        self.token2idx = {tok: i for i, tok in enumerate(self.vocab)}

    def dicts_to_csr(self, X_dicts):
        rows, cols, vals = [], [], []
        for i, d in enumerate(X_dicts):
            for token, cnt in d.items():
                j = self.token2idx.get(token)
                if j is not None:
                    rows.append(i); cols.append(j); vals.append(cnt)
        return csr_matrix((vals, (rows, cols)),
                          shape=(len(X_dicts), len(self.vocab)))

    def train(self, X_dicts, y):
        assert self.vocab is not None 
        Xs = self.dicts_to_csr(X_dicts)
        self.model.fit(Xs, y)

    def predict(self, word_count_dict):
        Xs    = self.dicts_to_csr([word_count_dict])
        pred  = self.model.predict(Xs)[0]
        proba = self.model.predict_proba(Xs)[0]
        return pred, {False: proba[0], True: proba[1]}














class NB:
    def __init__(self, TRAIN_SIZE):
        self.word_counts = {0: {}, 1: {}}  
        self.class_doc_counts = {0: 0, 1: 0}
        self.total_docs = 0
        self.vocab = set()



class NB:
    def __init__(self,TRAIN_SIZE):
        self.word_counts      = {0: defaultdict(int), 1: defaultdict(int)}
        self.class_doc_counts = {0: 0, 1: 0}
        self.total_docs       = 0
        self.vocab            = None
        self.total_words      = {}

    def train(self, X_dicts, y):
       
        assert self.vocab is not None

        
        for x_dict, label in zip(X_dicts, y):
            self.class_doc_counts[label] += 1
            self.total_docs += 1
            for token, cnt in x_dict.items():
                
                if token in self.vocab:
                    self.word_counts[label][token] += cnt


        for label in (0, 1):
            self.total_words[label] = sum(self.word_counts[label].values())

    def predict(self, x_dict):
        V = len(self.vocab)
        scores = {}
        for label in (0, 1):
            
            prior = (self.class_doc_counts[label] + 1) / (self.total_docs + 2)
            log_prob = math.log(prior)
            denom = self.total_words[label] + V

            
            for token, cnt in x_dict.items():
                
                wc = self.word_counts[label].get(token, 0) + 1
                log_prob += cnt * math.log(wc / denom)

            scores[label] = log_prob

        
        max_log = max(scores.values())
        exp_scores = {lbl: math.exp(scr - max_log) for lbl, scr in scores.items()}
        total = sum(exp_scores.values())
        probs = {lbl: esc / total for lbl, esc in exp_scores.items()}

        
        predicted = max(probs, key=probs.get)
        return predicted, probs










class loader:
    def __init__(self, folder_path): 
        self.folder_path = folder_path
        self.data = None   

    def load_data(self):
        csv_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.csv')]
        data_list = []
        for file in csv_files:
            full_path = os.path.join(self.folder_path, file)
            df = pd.read_csv(full_path)
            data_list.append(df)
        self.data = pd.concat(data_list, ignore_index=True)
        return self.data

    def split_data(self, TRAIN_SIZE):
        if self.data is None:
            self.preprocess_data()

        total = len(self.data)
        train_end = int(total * TRAIN_SIZE)
        test_start = int(total * 0.80)  #to always have 20%

        train_data = self.data.iloc[:train_end]
        test_data = self.data.iloc[test_start:]

        return train_data, test_data


    def preprocess_data(self):

        if self.data is None:
            self.load_data()

        self.data = self.data[['text', 'male']].dropna()
        self.data['male'] = self.data['male'].apply(lambda x: 1 if x else 0)
        

       


        

        def replace_emojis(text):
            return emoji.replace_emoji(text, replace=' <EMOJI_TOKEN> ')

        def clean_and_tokenize(text):
            text = html.unescape(text)
            text = re.sub(r"http\S+", " URL_TOKEN ", text)
            text = re.sub(r"\S+@\S+", " EMAIL_TOKEN ", text)
            text = re.sub(r"@\w+", " MENTION_TOKEN ", text)
            text = replace_emojis(text)
            text = text.lower()
            text = re.sub(r"[^a-z0-9_%<>\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text.split()



        self.data['text'] = self.data['text'].apply(clean_and_tokenize)
        return self.data







class Classifier:
    def __init__(self, algo_id, TRAIN_SIZE, vocab=None):
        self.algo_id = algo_id

        if algo_id == 1:
            self.ALGO = LR(TRAIN_SIZE)
            if vocab is not None:
                self.ALGO.set_vocab(list(vocab))
        else:
            self.ALGO = NB(TRAIN_SIZE)

        if vocab is not None:
            if algo_id == 0:               
                self.ALGO.vocab = set(vocab)
            else:                          
                self.ALGO.vocab = list(vocab)

    def train(self, X_train, y_train):
        self.ALGO.train(X_train, y_train)

    def predict_vector(self, vector):
        return self.ALGO.predict(vector)








class Vectorizer:
    def __init__(self, docs):
        self.vocab = self.build_vocab(docs)

    def build_vocab(self, docs):
        vocab = set()
        for doc in docs:
            for token in doc:              
                vocab.add(token)           
        return vocab


    def transform(self, doc):
        counts = {}
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def transform_batch(self, docs):
        return [self.transform(doc) for doc in docs]






def evaluate_metrics(y_true, y_pred):
    tp = sum((yt and yp) for yt, yp in zip(y_true, y_pred))
    tn = sum((not yt and not yp) for yt, yp in zip(y_true, y_pred))
    fp = sum((not yt and yp) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt and not yp) for yt, yp in zip(y_true, y_pred))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    accuracy = (tp + tn) / len(y_true)
    f_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0

    print("\nTest results / metrics:")
    print(f"Number of true positives: {tp}")
    print(f"Number of true negatives: {tn}")
    print(f"Number of false positives: {fp}")
    print(f"Number of false negatives: {fn}")
    print(f"Sensitivity (recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Negative predictive value: {npv:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F-score: {f_score:.4f}")
