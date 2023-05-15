import pandas as pd
import yaml
import argparse
import os
from transformers import BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.data import Dataset
import tensorflow as tf
import numpy as np 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from bert_text_classifier import BertTextClassifier  # your TensorFlow model script


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # load pretrained tokenizer model

class TFDataGenerator(Dataset):
    def __init__(self, texts, labels):
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="tf") for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, texts, labels, tokenizer, max_length, batch_size):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def __len__(self):
        return len(self.texts) // self.batch_size

    def __getitem__(self, index):
        batch_texts = self.texts[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        processed_data = self._process_data(batch_texts, batch_labels)
        return processed_data

    def _process_data(self, texts, labels):
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels


class BertTextClassifierTrain:
    
    def __init__(self, config_path):
        self.config_path = config_path

    def load_config(self):    
        with open(self.config_path, 'r') as stream:
            self.model_config = yaml.safe_load(stream)

    def load_model(self):
        self.model = BertTextClassifier(num_labels=len(self.model_config['labels'].values()))

    def load_data(self):
        df_path = os.path.dirname(os.path.abspath(self.config_path))+"/df.csv"
        self.df = pd.read_csv(df_path)
        self.df = self.df[["messageText", "cluster"]]
        self.df.columns = ["text", "category"]
        print(self.model_config['labels'].values())
        self.df = self.df[self.df["category"].isin(self.model_config['labels'].values())]
        self.df = self.df[self.df.text.str.len() < 512]
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        problematic_texts = []

        for text in self.df.text:  # assume `texts` is your list of texts
            encoded = tokenizer.encode(text, truncation=False)
            if max(encoded) >= 28996:
                problematic_texts.append(text)
        print("Problem", problematic_texts)
        # print out the problematic texts
        for text in problematic_texts:
            print("Problem", text)
        print("Data loaded, using {} samples".format(len(self.df)))
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.2, random_state=42)
        self.df_val, self.df_test = train_test_split(self.df_test, test_size=0.5, random_state=42)

    def train(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        train_texts = self.df_train['text'].tolist()
        train_labels = self.df_train['category'].tolist()
        val_texts = self.df_val['text'].tolist()
        val_labels = self.df_val['category'].tolist()

        # Tokenize and encode the input texts
        train_encodings = tokenizer.batch_encode_plus(
            train_texts,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        val_encodings = tokenizer.batch_encode_plus(
            val_texts,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        train_inputs = {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        }
        val_inputs = {
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask']
        }

        train_labels = tf.convert_to_tensor(train_labels)
        val_labels = tf.convert_to_tensor(val_labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).batch(self.model_config['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)).batch(self.model_config['batch_size'])

        optimizer = Adam(learning_rate=self.model_config['lr'])
        loss = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            verbose=1
        )

        self.model.save(os.path.dirname(os.path.abspath(self.config_path))+"/model")

    def evaluate(self):
        test = TFDataGenerator(self.df_test['text'].tolist(), self.df_test['category'].tolist())
        results = self.model.evaluate(test, verbose=0)
        print(f'Test Accuracy: {results[1]:.3f}')
    
    def run_all(self):  
        self.load_config()
        self.load_model()
        self.load_data()
        self.train()
        self.evaluate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()
    BertTextClassifierTrain_ = BertTextClassifierTrain(args.config_path)
    BertTextClassifierTrain_.run_all()

if __name__ == '__main__':
    main()
