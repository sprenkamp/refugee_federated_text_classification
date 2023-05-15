import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu

class BertTextClassifier(tf.keras.Model):
    def __init__(self, num_labels, dropout=0.5):
        super(BertTextClassifier, self).__init__()

        self.bert = TFBertModel.from_pretrained('bert-base-cased')
        self.dropout = Dropout(dropout)
        self.dense = Dense(768, activation='relu')
        self.final_dense = Dense(num_labels, activation='softmax')

    def call(self, inputs):
        input_id = inputs['input_ids']
        mask = inputs['attention_mask']
        bert_output = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = bert_output[1]
        dropout_output = self.dropout(pooled_output)
        dense_output = self.dense(dropout_output)
        final_output = self.final_dense(dense_output)
        return final_output
