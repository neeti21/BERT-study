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

def train(model, train_dataset, col1, col2, criteria, epochs = 3, batch_size = 32, lr = 3e-5, criterion = nn.CrossEntropyLoss()):
    '''
    Use train function to train the BERT models.
    '''

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    n_total_steps = len(train_loader)
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if col2:
            encoded_input = tokenizer(list(batch[col1]), list(batch[col2]), padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(device)
            else:
            encoded_input = tokenizer(list(batch[col1]), padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(device)
            input_ids = encoded_input.input_ids
            attention_mask = encoded_input.attention_mask
            labels = torch.tensor(batch['label']).type(torch.LongTensor) 
            labels = labels.to(device)

            #forward pass
            outputs = model(input_ids, attention_mask, criteria)

            #backward pass
            loss = criterion(outputs, labels)
            loss.backward()

            #weights optimization
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) %1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    return model