import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import pipeline
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import argparse
import os
import yaml

tokenizer = BertTokenizer.from_pretrained('bert-base-cased') #load pretrained tokenizer model

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # self.labels = [labels[label] for label in df['category']]
        self.texts_encoded = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['text']]
        self.texts_char = df["text"].to_list()
    # def classes(self):
    #     return self.labels

    def __len__(self):
        return len(self.texts_encoded)

    # def get_batch_labels(self, idx):
    #     # Fetch a batch of labels
    #     return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts_encoded[idx]
    
    def get_batch_texts_char(self, idx):
        # Fetch a batch of original texts
        return self.texts_char[idx]

    def __getitem__(self, idx):

        batch_texts_encoded = self.get_batch_texts(idx)
        batch_texts_char = self.get_batch_texts_char(idx)

        return batch_texts_encoded, batch_texts_char

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

class BertTextClassifierInference:
    
    def __init__(self, input_file_path, config_path):
        self.input_file_path = input_file_path
        self.config_path = config_path
    
    def load_config(self):    
        with open(self.config_path, 'r') as stream:
            self.model_config = yaml.safe_load(stream)
    
    def load_model(self):
        self.model = BertClassifier()
        self.model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(self.config_path))+"/model.pt"))
        print("model loaded")

    def set_device(self): #TODO nested if with CPU
         self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
         self.model.to(self.device)
         print("device")
    
    def load_data(self):
        self.df = pd.read_csv(self.input_file_path, names=['text'])
        print("load_data")


    def evaluate_unlabeled(self):
        print("evaluate")
        dataset_test = Dataset(self.df)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1)
        self.original_text_char_list = []
        self.category_pred_list = []
        with torch.no_grad():
            for input_test, original_text_char in tqdm(dataloader_test):
                mask = input_test['attention_mask'].to(self.device)
                input_id = input_test['input_ids'].squeeze(1).to(self.device)
                output = self.model(input_id, mask)
                self.original_text_char_list.append(original_text_char[0])
                self.category_pred_list.append(list(self.model_config['labels'].keys())[list(self.model_config['labels'].values()).index(output.argmax(dim=1).cpu().item())])
    
    def write_results(self):
        print("write")
        df_out = pd.DataFrame({"text":self.original_text_char_list,
                                "category_pred": self.category_pred_list})
        dataset_name_str = os.path.basename(self.input_file_path).split(".")[0]
        df_out.to_csv(f"{os.path.dirname(os.path.abspath(self.config_path))}/{dataset_name_str}.csv", index=None)
    
    def run_all(self):
        self.load_config()  
        self.load_model()
        self.set_device()
        self.load_data()
        self.evaluate_unlabeled()
        self.write_results()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', help="Specify the path of the input file, each line should contain a single line of text", required=True) #TODO change to argparse.FileType('r')
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()
    BertTextClassifierInferenceClass = BertTextClassifierInference(args.input_file_path, args.config_path)
    BertTextClassifierInferenceClass.run_all()

if __name__ == '__main__':
    main()
