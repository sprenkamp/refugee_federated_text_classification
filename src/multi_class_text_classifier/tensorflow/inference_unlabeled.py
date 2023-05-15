import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
from tqdm import tqdm
import argparse
import os
import yaml

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # load pretrained tokenizer model

class TFDataset:
    def __init__(self, df):
        self.texts_encoded = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="tf") for text in df['text']]
        self.texts_char = df["text"].tolist()

    def __len__(self):
        return len(self.texts_encoded)

    def __getitem__(self, idx):
        return self.texts_encoded[idx], self.texts_char[idx]

class BertTextClassifierInference:
    
    def __init__(self, input_file_path, config_path):
        self.input_file_path = input_file_path
        self.config_path = config_path
    
    def load_config(self):    
        with open(self.config_path, 'r') as stream:
            self.model_config = yaml.safe_load(stream)
    
    def load_model(self):
        self.model = load_model(os.path.dirname(os.path.abspath(self.config_path))+"/model")
        print("Model loaded")
    
    def load_data(self):
        self.df = pd.read_csv(self.input_file_path, names=['text'])
        print("Data loaded")

    def evaluate_unlabeled(self):
        print("Evaluating")
        dataset_test = TFDataset(self.df)
        self.original_text_char_list = []
        self.category_pred_list = []
        for input_test, original_text_char in tqdm(dataset_test):
            input_ids = input_test['input_ids']
            attention_mask = input_test['attention_mask']
            output = self.model([input_ids, attention_mask])
            self.original_text_char_list.append(original_text_char[0])
            self.category_pred_list.append(list(self.model_config['labels'].keys())[list(self.model_config['labels'].values()).index(np.argmax(output))])
    
    def write_results(self):
        print("Writing results")
        df_out = pd.DataFrame({"text": self.original_text_char_list,
                                "category_pred": self.category_pred_list})
        dataset_name_str = os.path.basename(self.input_file_path).split(".")[0]
        df_out.to_csv(f"{os.path.dirname(os.path.abspath(self.config_path))}/{dataset_name_str}.csv", index=None)
    
    def run_all(self):
        self.load_config()  
        self.load_model()
        self.load_data()
        self.evaluate_unlabeled()
        self.write_results()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', help="Specify the path of the input file, each line should contain a single line of text", required=True)
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()
    BertTextClassifierInferenceClass = BertTextClassifierInference(args.input_file_path, args.config_path)
    BertTextClassifierInferenceClass.run_all()

if __name__ == '__main__':
    main()
