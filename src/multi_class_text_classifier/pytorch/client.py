import torch
import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score



class CustomTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, dropout=0.5):
        super(CustomTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_id, mask):
        embedded = self.embedding(input_id)
        embedded = self.dropout(embedded)
        hidden = self.relu(self.linear1(embedded.mean(dim=1)))
        output = self.linear2(hidden)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Flower client
class BertClient(fl.client.NumPyClient):
    def __init__(self, country, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # Determine vocab_size
        vocab_size = len(self.tokenizer.vocab)

        # Determine embedding_dim and hidden_dim
        embedding_dim = 300  # Adjust this value as needed
        hidden_dim = 128  # Adjust this value as needed

        self.model = CustomTextClassifier(vocab_size, embedding_dim, hidden_dim, num_labels=num_labels)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = nn.CrossEntropyLoss()
        self.country = country
        self.load_data(country=self.country)

    def get_parameters(self, config=None):
        # Detach each parameter tensor before converting to a numpy array
        return [np.asarray(p.cpu().detach(), dtype='float32') for p in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def load_data(self, country):
        self.df = pd.read_csv("/home/ubuntu/refugee_supervised_text_classification/models/firstTry/df_testing.csv")
        self.df = self.df[self.df.x.str.len() < 512]
        self.df = self.df[self.df["federation_level"] == country]
        print("Data loaded, using {} samples".format(len(self.df)))
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.2, random_state=42)
        self.df_val, self.df_test = train_test_split(self.df_test, test_size=0.5, random_state=42)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        tokenizer = self.tokenizer
        train_texts = self.df_train['x'].tolist()
        train_labels = self.df_train['y'].tolist()

        # Tokenize and encode the input texts
        train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

        train_inputs = train_encodings['input_ids'].to(device)
        train_attention = train_encodings['attention_mask'].to(device)
        train_labels = torch.tensor(train_labels).to(device)

        train_dataset = TensorDataset(train_inputs, train_attention, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Set model to training
        self.model.train()
        for epoch in range(3):
            for _, data in enumerate(train_loader, 0):
                ids = data[0]
                mask = data[1]
                targets = data[2]
                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.long)

                outputs = self.model(ids, mask)
                self.optimizer.zero_grad()
                loss = self.loss(outputs, targets)
                if _ % 5000 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()

        accuracy, _, metrics = self.evaluate(parameters, config)

        # Calculate additional metrics
        true_labels = self.df_train['y'].tolist()
        predicted_labels = [torch.max(self.model(ids.to(device), mask.to(device)), dim=1)[1].cpu().tolist()
                            for ids, mask, _ in train_loader]
        predicted_labels = [item for sublist in predicted_labels for item in sublist]

        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1) #TODO change zero devision
        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
        f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)

        num_examples = len(self.df_train['x'])
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        print("fit", self.country, metrics)

        return self.get_parameters(), num_examples, metrics



    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        tokenizer = self.tokenizer
        test_texts = self.df_test['x'].tolist()
        test_labels = self.df_test['y'].tolist()

        # Tokenize and encode the test texts
        test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

        test_inputs = test_encodings['input_ids'].to(device)
        test_attention = test_encodings['attention_mask'].to(device)
        test_labels = torch.tensor(test_labels).to(device)

        test_dataset = TensorDataset(test_inputs, test_attention, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=4)

        # Set model to evaluation mode
        self.model.eval()
        total = 0
        correct = 0
        loss = 0.0
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for _, data in enumerate(test_loader, 0):
                ids = data[0]
                mask = data[1]
                targets = data[2]

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.long)

                outputs = self.model(ids, mask)
                _, predicted = torch.max(outputs.data, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                batch_loss = self.loss(outputs, targets)
                loss += batch_loss.item() * targets.size(0)  # Accumulate the batch loss

                predicted_labels.extend(predicted.cpu().tolist())
                true_labels.extend(targets.cpu().tolist())

        accuracy = correct / total
        average_loss = loss / total

        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1) #TODO change zero devision
        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
        f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)


        num_examples = len(test_texts)
        metrics = {
            'loss': average_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        aggregated_model = self.model.state_dict()
        torch.save(aggregated_model, "aggregated_model_state_dict.pth")

        print("eval", self.country, metrics)

        return accuracy, num_examples, metrics


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=BertClient(country=sys.argv[1], num_labels=int(sys.argv[2])))
