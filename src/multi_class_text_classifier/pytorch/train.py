import pandas as pd
import yaml
import torch
import argparse
import os
from transformers import BertTokenizer
from bert_text_classifier import BertTextClassifier
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from tqdm import tqdm



tokenizer = BertTokenizer.from_pretrained('bert-base-cased') #load pretrained tokenizer model

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels):
        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertTextClassifierTrain:
    
    def __init__(self, config_path):
        self.config_path = config_path

    def load_config(self):    
        with open(self.config_path, 'r') as stream:
            self.model_config = yaml.safe_load(stream)
            # print(self.model_config)
    
    def load_model(self):
        self.model = BertTextClassifier()

    def set_device(self): #TODO nested if with CPU
         self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
         self.model.to(self.device)
    
    def load_data(self):
        df_path = os.path.dirname(os.path.abspath(self.config_path))+"/df.csv"
        self.df = pd.read_csv(df_path, names=['text','category'])
        np.random.seed(112)
        self.df_train, self.df_val, self.df_test = np.split(self.df.sample(frac=1, random_state=42), 
                                     [int(.8*len(self.df)), int(.9*len(self.df))])
        self.df_train.to_csv(os.path.dirname(os.path.abspath(self.config_path))+"/df_train.csv", index=None, header=None)
        self.df_val.to_csv(os.path.dirname(os.path.abspath(self.config_path))+"/df_val.csv", index=None, header=None)
        self.df_test.to_csv(os.path.dirname(os.path.abspath(self.config_path))+"/df_test.csv", index=None, header=None)

    def train(self):     
        train, val = Dataset(df=self.df_train, labels=self.model_config['labels']), Dataset(df=self.df_val, labels=self.model_config['labels'])

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=self.model_config['batch_size'], shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=self.model_config['batch_size'])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.model_config['lr']))

        for epoch_num in range(self.model_config['epochs']):

                total_acc_train = 0
                total_loss_train = 0

                for train_input, train_label in tqdm(train_dataloader):

                    train_label = train_label.to(self.device)
                    mask = train_input['attention_mask'].to(self.device)
                    input_id = train_input['input_ids'].squeeze(1).to(self.device)
                    self.model.to(self.device)

                    output = self.model(input_id, mask)
                    
                    batch_loss = criterion(output, train_label.long())
                    total_loss_train += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc

                    self.model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                
                total_acc_val = 0
                total_loss_val = 0

                with torch.no_grad():

                    for val_input, val_label in val_dataloader:

                        val_label = val_label.to(self.device)
                        mask = val_input['attention_mask'].to(self.device)
                        input_id = val_input['input_ids'].squeeze(1).to(self.device)
                        output = self.model(input_id, mask)

                        batch_loss = criterion(output, val_label.long())
                        total_loss_val += batch_loss.item()
                        
                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc
                
                print(f'Epochs: {epoch_num + 1} | \
                        Train Loss: {total_loss_train / len(self.df_train): .3f} | \
                        Train Accuracy: {total_acc_train / len(self.df_train): .3f} | \
                        Val Loss: {total_loss_val / len(self.df_val): .3f} | \
                        Val Accuracy: {total_acc_val / len(self.df_val): .3f}')

        torch.save(self.model.state_dict(), os.path.dirname(os.path.abspath(self.config_path))+"/model.pt")
    
    def evaluate(self):

        test = Dataset(self.df_test)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

        total_acc_test = 0

        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(self.device)
                mask = test_input['attention_mask'].to(self.device)
                input_id = test_input['input_ids'].squeeze(1).to(self.device)
                output = self.model(input_id, mask)
                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        print(f'Test Accuracy: {total_acc_test / len(self.df_test): .3f}')
    
    def run_all(self):  
        self.load_config()
        self.load_model()
        self.set_device()
        self.load_data()
        self.train()
        self.evaluate()
        # self.write_results()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()
    BertTextClassifierTrain_ = BertTextClassifierTrain(args.config_path)
    BertTextClassifierTrain_.run_all()

if __name__ == '__main__':
    main()
