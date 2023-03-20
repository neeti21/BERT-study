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

def accuracy(model, test_dataset, col1, col2, criteria):
    '''
    Use this function to evaluate accuracy of the model.
    If only one text column is present, then keep col2 = None.
    '''

    from torchmetrics import Accuracy

    sent1, sent2 = list(test_dataset[col1]), list(test_dataset[col2])
    y_true_label = torch.tensor(list(test_dataset['label']))

    pred_test = []
    with torch.no_grad():
    for i in range(0, len(sent1)):
        if col2:
        encoded_input = tokenizer(sent1[i], sent2[i], padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(device)
        else:
        encoded_input = tokenizer(sent1[i], padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(device)
        input_ids = encoded_input.input_ids
        attention_mask = encoded_input.attention_mask
        y_predicted = model(input_ids, attention_mask, criteria)
        y_predicted_class = torch.max(y_predicted, 1)[1]
        pred_test.append(y_predicted_class.item())

    acc = Accuracy()
    acc_value = acc(torch.tensor(pred_test), y_true_label)
    return acc_value.item()

def f1(model, test_dataset, col1, col2, criteria):
    '''
    Use this function to evaluate f1 score of the model.
    If only one text column is present, then keep col2 = None.
    '''
    from torchmetrics.classification import BinaryF1Score

    sent1, sent2 = list(test_dataset[col1]), list(test_dataset[col2])
    y_true_label = torch.tensor(list(test_dataset['label']))

    pred_test = []
    with torch.no_grad():
    for i in range(0, len(sent1)):
        if col2:
        encoded_input = tokenizer(sent1[i], sent2[i], padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(device)
        else:
        encoded_input = tokenizer(sent1[i], padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(device)
        input_ids = encoded_input.input_ids
        attention_mask = encoded_input.attention_mask
        y_predicted = model(input_ids, attention_mask, criteria)
        y_predicted_class = torch.max(y_predicted, 1)[1]
        pred_test.append(y_predicted_class.item())

    f1 = BinaryF1Score()
    f1_value = f1(torch.tensor(pred_test), y_true_label)
    return f1_value.item()