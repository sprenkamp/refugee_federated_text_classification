import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import pipeline
from torch import nn
from torch.optim import Adam


class BertTextClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertTextClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output) #TODO change to softmax (possibly linear, sigmoid)

        return final_layer
