import os
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
import numpy as np
import math

class LR:
    def __init__(self, TRAIN_SIZE):
        self.TRAIN_SIZE = TRAIN_SIZE
        self.model = LogisticRegression(max_iter=1000)
        self.vocab = None

    def _dict_to_vector(self, word_count_dict):
        # Convert dict to list (vector) ordered by vocab
        return [word_count_dict[word] for word in self.vocab]

    def train(self, X_dicts, y):
        self.vocab = list(X_dicts[0].keys())
        X = np.array([self._dict_to_vector(d) for d in X_dicts])
        self.model.fit(X, y)

    def predict(self, word_count_dict):
        X = np.array([self._dict_to_vector(word_count_dict)])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return pred, {False: proba[0], True: proba[1]}













class NB:
    def __init__(self, TRAIN_SIZE):
        self.word_counts = {0: {}, 1: {}}  
        self.class_doc_counts = {0: 0, 1: 0}
        self.total_docs = 0
        self.vocab = set()

    def train(self, X_dicts, y):
        for x, label in zip(X_dicts, y):
            self.class_doc_counts[label] += 1
            self.total_docs += 1

            for word, count in x.items():
                self.vocab.add(word)
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0
                self.word_counts[label][word] += count

    def predict(self, x_dict):
        vocab_size = len(self.vocab)
        scores = {}

        for label in [0, 1]:
            log_prob = math.log(self.class_doc_counts[label] / self.total_docs)
            total_words_in_class = sum(self.word_counts[label].values())

            for word, count in x_dict.items():
                word_count = self.word_counts[label].get(word, 0)
                word_prob = (word_count + 1) / (total_words_in_class + vocab_size)
                log_prob += count * math.log(word_prob)

            scores[label] = log_prob

        # Normalize scores
        max_log = max(scores.values())
        exp_scores = {k: math.exp(v - max_log) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}

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


    def preprocess_data(self):##to change
        if self.data is None:
            self.load_data()

        self.data = self.data[['text', 'male']].dropna()# keep only the features we need 

        self.data['male'] = self.data['male'].apply(lambda x: 1 if x else 0)# from boolean to int

        def clean_and_tokenize(text):
            text = text.lower()
            text = re.sub(r"[^a-z0-9\s]", '', text)
            return text.split()

        self.data['text'] = self.data['text'].apply(clean_and_tokenize)

        return self.data







class Classifier:

    def __init__(self, algo_id, TRAIN_SIZE, train_data):
        self.algo_id = algo_id
        if algo_id == 1:
            self.ALGO = LR(TRAIN_SIZE)
        else:
            self.ALGO = NB(TRAIN_SIZE)
        self.train_data = train_data

    def train(self):
        self.ALGO.train(self.train_data)

    def predict(self, sentence):
        return self.ALGO.predict(sentence)






class Vectorizer:
    def __init__(self, docs):
        self.vocab = self.build_vocab(docs)

    def build_vocab(self, docs):
        vocab = set()
        for doc in docs:
            for tokens in doc:
                vocab.update(tokens)
        return sorted(list(vocab))

    def transform(self, doc):
        token_counts = dict.fromkeys(self.vocab, 0)
        for token in doc:
            if token in token_counts:
                token_counts[token] += 1
        return token_counts

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
