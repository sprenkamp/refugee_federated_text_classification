import torch
import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch import nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the PyTorch model and optimizer
num_labels = 3  # Number of classes: medical_info, transportation, asylum
model = BertForSequenceClassification.from_pretrained('fine_tuned_model', num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Define a class that inherits from the flwr.client.NumPyClient class
class PyTorchClient(fl.client.NumPyClient):
    def __init__(self, country):
        self.load_data(country=country)    
    
    def get_parameters(self):
        return [val.cpu().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):
        parameters = [torch.Tensor(val) for val in parameters]
        model.load_state_dict(zip(model.state_dict().keys(), parameters))
        optimizer.zero_grad()

    def load_and_preproccess_data(self, country):
        # Load data from CSV file
        df = pd.read_csv('models/firstTry/df_dummy.csv', on_bad_lines="skip")
        df = df[self.df.x.str.len() < 512]
        df = df[self.df["federation_level"] == country]
        print("Data loaded, using {} samples from {}".format(len(self.df),country))

        # Split the df into train and validation sets
        train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

        # Tokenize and encode the text data for train and validation sets
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train_inputs = tokenizer(
            train_data['x'].tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        val_inputs = tokenizer(
            val_data['x'].tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Convert labels to tensors for train and validation sets
        train_labels = torch.tensor(train_data['y'].tolist())
        val_labels = torch.tensor(val_data['y'].tolist())

        # Create TensorDatasets for train and validation sets
        train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
        val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

        # Define batch size and create DataLoaders for train and validation sets
        batch_size = 8
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Perform local training on the client
        total_loss = 0
        model.train()

        for batch in self.train_dataloader:
            input_ids, attention_mask, batch_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(self.train_dataloader)

        # Return the updated parameters and the number of training examples
        return self.get_parameters(config={}), len(self.train_dataloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Perform local evaluation on the client
        model.eval()
        total_loss = 0
        total_correct = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Calculate the number of correct predictions
                _, predicted_labels = torch.max(logits, dim=1)
                total_correct += torch.sum(predicted_labels == batch_labels).item()

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / len(self.val_dataloader.dataset)

        # Return the evaluation result
        return float(loss), len(self.val_dataloader.dataset), {"accuracy": float(accuracy)}

# Create an instance of the PyTorchClient class
client = PyTorchClient()

# Start the client and connect it to the Flower server
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PyTorchClient(country=sys.argv[1], num_labels=int(sys.argv[2])))

