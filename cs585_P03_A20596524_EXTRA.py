import os
import pandas as pd

class LR:
    def __init__(self, TRAIN_SIZE):
        self.TRAIN_SIZE = TRAIN_SIZE

class NB:
    def __init__(self, TRAIN_SIZE):
        self.TRAIN_SIZE = TRAIN_SIZE

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
        .
        if self.data is None:
            self.load_data()
        train_count = int(len(self.data) * TRAIN_SIZE)
        train_data = self.data.iloc[:train_count]
        test_data = self.data.iloc[train_count:]
        print("Splitting the data...")
        return train_data, test_data

    def export_csv(self, df, output_filename):
        
        df.to_csv(output_filename, index=False)
        print(f"CSV exported to {output_filename}")

    def preprocess_data(self):
        # Implement any data cleanup or wrangling necessary.
        # For example, you could drop missing values, lower case the text, etc.
        if self.data is None:
            self.load_data()
        # Example preprocessing step:
        # self.data.dropna(inplace=True)
        # self.data['text'] = self.data['text'].str.lower()
        print("Preprocessing the data...")
        return self.data

class Classifier:
    def __init__(self, algo_id, TRAIN_SIZE, data):
        if algo_id == 1:
            self.ALGO = LR(TRAIN_SIZE)
        else:
            self.ALGO = NB(TRAIN_SIZE)
        self.data = data

    def train(self):
        print("Training the classifier...")

    def predict(self, input_data):
        print("Predicting with the classifier...")
        return "Prediction"
