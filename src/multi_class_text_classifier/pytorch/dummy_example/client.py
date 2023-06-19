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
import grpc
from collections import OrderedDict

# Set the maximum message size (e.g., 1 GB)
max_message_size = 1024 * 1024 * 1024
channel = grpc.insecure_channel('127.0.0.1:8080', options=[
    ('grpc.max_receive_message_length', max_message_size),
    ('grpc.max_send_message_length', max_message_size)
])

# max_message_length = 1024 * 1024 * 1024  # 1 GB
# server = grpc.server(
#     futures.ThreadPoolExecutor(max_workers=10),
#     options=[
#         ('grpc.max_send_message_length', max_message_length),
#         ('grpc.max_receive_message_length', max_message_length),
#     ],
# )


# Define a class that inherits from the flwr.client.NumPyClient class
class PyTorchClient(fl.client.NumPyClient):
    def __init__(self, country):
        self.country = country
        self.load_and_preproccess_data()    
        num_labels = 5  # Number of classes: medical_info, transportation, asylum
        self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=num_labels)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)


    # def get_parameters(self, config=None):
    #     # Detach each parameter tensor before converting to a numpy array
    #     return [np.asarray(p.cpu().detach(), dtype='float32') for p in self.model.parameters()]
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # def set_parameters(self, parameters):
    #     params_dict = zip(self.model.state_dict().keys(), parameters)
    #     # print(params_dict.shape)
    #     state_dict = {k: torch.Tensor(v) for k, v in params_dict}
    #     # print(state_dict.shape)
    #     for param_tensor in self.model.state_dict():    
    #         print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
    #     self.model.load_state_dict(state_dict, strict=True)

    def load_and_preproccess_data(self):
        # Load data from CSV file
        self.df = pd.read_csv('data/df_shuffled_sven.csv', on_bad_lines="skip")
        self.df = self.df[self.df.x.str.len() < 512]
        self.df = self.df[self.df["federation_level"] == self.country]
        print("Data loaded, using {} samples from {}".format(len(self.df),self.country))

        # Split the df into train and validation sets
        train_data, val_data = train_test_split(self.df, test_size=0.2, random_state=42)

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
        self.train_labels = torch.tensor(train_data['y'].tolist())
        self.val_labels = torch.tensor(val_data['y'].tolist())

        # Create TensorDatasets for train and validation sets
        train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], self.train_labels)
        val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], self.val_labels)

        # Define batch size and create DataLoaders for train and validation sets
        batch_size = 8
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        num_epochs = 3  # Set the number of epochs

        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()

            for batch in self.train_dataloader:
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Country {self.country} Epoch {epoch+1} - Average Loss: {avg_loss}")

            loss, _, metrics = self.evaluate(self.get_parameters(config), config)

        # Return the updated parameters and the number of training examples
        return self.get_parameters(config={}), len(self.train_dataloader.dataset), metrics

    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Perform local evaluation on the client
        self.model.eval()
        total_loss = 0
        total_correct = 0

        predicted_labels = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Calculate the number of correct predictions
                _, batch_predicted_labels = torch.max(logits, dim=1)
                total_correct += torch.sum(batch_predicted_labels == batch_labels).item()

                predicted_labels.extend(batch_predicted_labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / len(self.val_dataloader.dataset)

        # Calculate additional metrics
        precision = precision_score(self.val_labels, predicted_labels, average='macro')
        recall = recall_score(self.val_labels, predicted_labels, average='macro')
        f1 = f1_score(self.val_labels, predicted_labels, average='macro')

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Return the evaluation result
        return float(loss), len(self.val_dataloader.dataset), metrics

# # Create an instance of the PyTorchClient class
# client = PyTorchClient()

# Start the client and connect it to the Flower server
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PyTorchClient(country=sys.argv[1]), grpc_max_message_length=1438900533)

