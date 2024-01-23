import pandas as pd
import torch 
import numpy as np
import ast
import torch_geometric
from ablang.tokenizers import ABtokenizer
from modules import AbLangGVP, AbLangGAT, AbLangLinear, AbLangSGNN
import deepfrier_utils
import data_loaders
import data_loaders
from kornia.losses.focal import FocalLoss
from torch.optim import lr_scheduler as lrs
from sklearn.utils.class_weight import compute_class_weight
from ablang.tokenizers import ABtokenizer
from torch.nn.functional import one_hot
from ablang import Pretrained
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from scipy.stats import spearmanr
from warmup_scheduler_pytorch import WarmUpScheduler
from torch.utils.data import Subset
from sklearn.metrics import r2_score
import random
import argparse
from train_ablang_gnn import TrainAbLangGNN
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from  matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset to train on")
parser.add_argument("-he", "--head", help="Graph head to use")
parser.add_argument("-k", "--k", help="Number of CV folds/model to ensemble")

args = parser.parse_args()
dataset = args.dataset
head = args.head
k = args.k

#hardcoded 
valid_size = {'tm':73,'psr':65,'hic-rt':86,'agg':305}

model_params = 'outputs/best_models/params/' + dataset + '_' + head
with open(model_params,'r') as fh:
    lines = fh.readlines()
    average_score = float(lines[0].split()[1])
    params = ast.literal_eval(lines[1])

predictions = torch.zeros((k, valid_size[dataset], params['outputs']))
true_labels = torch.zeros(valid_size[dataset], 1)  
model_path = 'outputs/best_models/' + dataset + '_' + head + '/'

for i in range(k):
    fold_path = model_path + 'fold' + str(i) + '.pth'
    trainer = TrainAbLangGNN(params)
    c = params['chains']
    b_s = params['batch_size']
    if head == 'linear':
        train_loader, valid_loader, train_data = trainer.load_sequence_datasets(c)
    else:
        train_loader, valid_loader, train_data = trainer.load_structure_datasets(c, ensemble = True) 
    model = trainer.init_model(params, head, train_dataset = train_data) if head == 'gvp' else trainer.init_model(params, head)
    state_dict = torch.load(fold_path)
    model.load_state_dict(state_dict, strict = True)
    model.eval()
    ctr = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            outputs = model(inputs).cpu()
            try:
                predictions[i, ctr*b_s:ctr*b_s + b_s, :] = outputs.reshape(predictions[i, ctr*b_s:ctr*b_s + b_s, :].shape)
                true_labels[ctr*b_s:ctr*b_s + b_s, :] = labels.reshape(true_labels[ctr*b_s:ctr*b_s + b_s, :].shape)
            except:
                predictions[i, ctr*b_s:ctr*b_s + len(outputs), :] = outputs.reshape(predictions[i, ctr*b_s:ctr*b_s + b_s, :].shape)
                true_labels[ctr*b_s:ctr*b_s+len(outputs), :] = labels.reshape(true_labels[ctr*b_s:ctr*b_s+len(outputs), :].shape)
            ctr+=1

avg_predictions = predictions.mean(dim=0)
predicted = torch.max(avg_predictions, 1)[1]
if dataset == 'agg':
    auroc = roc_auc_score(true_labels, avg_predictions[:,1])
    precision = precision_score(true_labels, predicted, average = 'binary', pos_label=1)
    recall = recall_score(true_labels, predicted, average = 'binary', pos_label=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    scores = [accuracy_score(torch.max(predictions[i,:,:], 1)[1], true_labels) for i in range(k)]
    ensemble_score = accuracy_score(predicted, true_labels)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 Score: {}".format(f1))
    print('AUROC Score: {}'.format(auroc))
else:
    ensemble_score = spearmanr(avg_predictions, true_labels).correlation
    plot_pred(avg_predictions, true_labels, dataset)
    scores = [spearmanr(predictions[i,:], true_labels).correlation for i in range(k)]

avg_holdout_score = sum(scores)/k
st_dev = np.std(scores)
st_err = 1.96*st_dev/np.sqrt(k)
    
print("Average Score: {}".format(average_score))
print("Ensemble Score: {}".format(ensemble_score))
print("Average Holdout Score: {} +- {}".format(avg_holdout_score, st_err))


'''
Copyright © 2023 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
