import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import requests
from tqdm import tqdm
import logging
import math
logger = logging.getLogger(__name__)
import urllib
import zipfile
from transformers import BertTokenizer, BertModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model = bert_model.to(device)

class Bert_NeuralNet(nn.Module):
    '''
    This class containes BERT model. 
    Use freeze_bert = True to ignore fine-tune of BERT layers. Only trains last Neural Network Linear layer.
    Use freeze_bert = False to train fine-tune BERT layers and train last Neural Network Linear layer.
    '''
    def __init__(self, freeze_bert = False, output_n = 2):
        super(Bert_NeuralNet, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        if freeze_bert:
          for p in self.bert_layer.parameters():
            p.requires_grad = False

        self.linear = nn.Linear(768, output_n)

    def forward(self, input_ids, attention_mask, criteria):
        token_embeddings = self.bert_layer(input_ids, attention_mask = attention_mask).last_hidden_state
        if criteria == "CLS":
          bert_emb = token_embeddings[:,0,:]
        elif criteria == "avg":
          bert_emb = token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
        elif criteria == "max":
          bert_emb, _ = torch.max((token_embeddings * attention_mask.unsqueeze(-1)), axis=1)
        y_pred = self.linear(bert_emb)
        return y_pred

class Bert_AttentionNN(nn.Module):
    '''
    This class containes BERT + Attention layer. 
    '''
    def __init__(self, output_n = 2):
        super(Bert_AttentionNN, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.attention_layer = nn.MultiheadAttention(768, 8)
        self.linear = nn.Linear(768, output_n)

    def forward(self, input_ids, attention_mask, criteria):
        token_embeddings = self.bert_layer(input_ids, attention_mask = attention_mask).last_hidden_state
        if criteria == "CLS":
          bert_emb = token_embeddings[:,0,:]
        elif criteria == "avg":
          bert_emb = token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
        elif criteria == "max":
          bert_emb, _ = torch.max((token_embeddings * attention_mask.unsqueeze(-1)), axis=1)
        bert_emb1, _ = self.attention_layer(query = bert_emb, key = bert_emb, value = bert_emb)
        y_pred = self.linear(bert_emb1)
        return y_pred

class Bert_CNN(nn.Module):
    '''
    This class containes BERT + CNN layer. 
    '''
    def __init__(self, output_n = 2):
        super(Bert_CNN, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.cnn = nn.Conv1d(1, 1, 379, stride = 1)
        self.cnn2 = nn.Conv1d(1, 1, 379, stride = 1)
        self.linear = nn.Linear(12, output_n)

    def forward(self, input_ids, attention_mask, criteria):
        token_embeddings = self.bert_layer(input_ids, attention_mask = attention_mask).last_hidden_state
        if criteria == "CLS":
          bert_emb = token_embeddings[:,0,:]
        elif criteria == "avg":
          bert_emb = token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
        elif criteria == "max":
          bert_emb, _ = torch.max((token_embeddings * attention_mask.unsqueeze(-1)), axis=1)
        print("bert shape: ", bert_emb.shape)
        bert_emb = torch.reshape(bert_emb, (bert_emb.shape[0], 1, bert_emb.shape[1]))
        print("CNN input: ", bert_emb.shape)
        cnn_output = self.cnn(bert_emb)
        print("CNN1 output: ", cnn_output.shape)
        cnn2_output = self.cnn2(cnn_output)
        print("CNN2 output: ", cnn2_output.shape)
        cnn2_output = torch.reshape(cnn2_output, (cnn2_output.shape[0], cnn2_output.shape[2]))
        y_pred = self.linear(cnn2_output)
        return y_pred