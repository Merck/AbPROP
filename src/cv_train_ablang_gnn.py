#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import StandardScaler
import torch_geometric
import torch
import pandas as pd
import numpy as np
from ablang.tokenizers import ABtokenizer
from modules import AbLangGVP, AbLangGAT, AbLangLinear, AbLangSGNN
import deepfrier_utils
import data_loaders
from kornia.losses.focal import FocalLoss
from torch.optim import lr_scheduler as lrs
from sklearn.utils.class_weight import compute_class_weight
from ablang.tokenizers import ABtokenizer
from torch.nn.functional import one_hot
from ablang import Pretrained
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from scipy.stats import spearmanr
from warmup_scheduler_pytorch import WarmUpScheduler
from torch.utils.data import Subset
from sklearn.metrics import r2_score
import random

class CVTrainAbLangGNN:
    
    def __init__(self, train_params):
        super().__init__()
        self.train_params = train_params
        self.parallel = False
        
        
    class SequenceDataset(Dataset):
        def __init__(self, X, Y):
            # store the inputs and outputs
            self.X = X
            self.y = Y
     
        # number of rows in the dataset
        def __len__(self):
            return len(self.X)
     
        # get a row at an index
        def __getitem__(self, idx):
            return [self.X[idx], self.y[idx]]

    def init_gvp_model(self, train_params, datum, max_length):
        node_in_dim = (datum.node_s.shape[1], datum.node_v.shape[1])
        node_h_dim = (256, 16)
        edge_in_dim = (datum.edge_s.shape[1], datum.edge_v.shape[1])
        edge_h_dim = (32, 1)
        
        model = AbLangGVP(
                node_in_dim=node_in_dim,
                node_h_dim=node_h_dim,
                edge_in_dim=edge_in_dim,
                edge_h_dim=edge_h_dim,
                residual = True,
                train_params = train_params,
                max_length = max_length
        )
        if train_params['pretrained']:
            print("Loading Pretrained GVP Weights")
            state_dict = torch.load('pretrained_weights/mifgvp.pt')['model_state_dict']
            #changing names of layers to match and removing mlm predicition head
            state_dict.pop('decoder.weight')
            state_dict.pop('decoder.bias')
            state_dict_fixed = {key.split('encoder.')[1]:val for key, val in state_dict.items()}
            model.load_state_dict(state_dict_fixed, strict = False)
        return model
    
    def init_model(self, train_params, train_dataset, model_choice):
        max_length_dict = {'psr':[147,132],'agg':[137], 'hic-rt':[147,132], 'tm':[147,132]}
        max_length = max_length_dict[train_params['dataset']]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_choice == 'gvp':
            datum = train_dataset[0][0]
            model = self.init_gvp_model(train_params = train_params, datum  = datum, max_length = max_length).to(device)
        elif model_choice == 'gat':
            model = AbLangGAT(train_params, max_length).to(device)
        elif model_choice == 'linear':
            model = AbLangLinear(train_params, max_length).to(device)
        elif model_choice == 'mifst':
            model = AbLangSGNN(train_params, max_length).to(device)
        return model
        
    def parallelize(self, model):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(device)
        self.parallel = True
        return model
    
    def freeze_encoder_l(self, model, n_layers):
        if n_layers > 12:
            print("Max value for n_layers is 12!")
            return
        elif n_layers == -1:
            for param in model.AbLangLight.AbRep.parameters():
                param.requires_grad = False
        else:
            for param in model.AbLangLight.AbRep.AbEmbeddings.parameters():
                param.requires_grad = False
            for layer in model.AbLangLight.AbRep.EncoderBlocks.Layers[:n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def freeze_encoder_h(self, model, n_layers):
        if n_layers > 12:
            print("Max value for n_layers is 12!")
            return
        elif n_layers == -1:
            for param in model.AbLangHeavy.AbRep.parameters():
                param.requires_grad = False
        else:
            for param in model.AbLangHeavy.AbRep.AbEmbeddings.parameters():
                param.requires_grad = False
            for layer in model.AbLangHeavy.AbRep.EncoderBlocks.Layers[:n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False      
                    
    def freeze_encoder(self, model, n_layers):
        if n_layers > 12:
            print("Max value for n_layers is 12!")
            return
        elif n_layers == -1:
            for param in model.AbLang.AbRep.parameters():
                param.requires_grad = False
        else:
            for param in model.AbLang.AbRep.AbEmbeddings.parameters():
                param.requires_grad = False
            for layer in model.AbLang.AbRep.EncoderBlocks.Layers[:n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def load_structure_datasets(self, chains, fold):
        vocab_path = 'ablang/vocab.json'
        tokenizer = ABtokenizer(vocab_path)
        mifst = self.train_params['model_choice'] == 'mifst'
        data = data_loaders.get_dataset(
            self.train_params['dataset'], chains,
            "train",tokenizer, self.train_params['top_k'], normalize = self.train_params['normalize'], mifst = mifst
        )
        
        random.seed(1)
        data = random.sample([x for x in data], len(data))   
        num_folds = self.train_params['num_folds']
        train_size = len(data)
        fold_size = train_size // num_folds
        valid_ind = np.zeros(train_size)
        
        if fold != (num_folds - 1):
            valid_ind[fold*fold_size:(fold+1)*fold_size] = 1
        else:
            valid_ind[fold*fold_size:] = 1
        
        valid_data, train_data = [data[int(x)] for x in np.where(valid_ind == 1)[0]], [data[int(x)] for x in np.where(valid_ind == 0)[0]]
        
        
        DataLoader = torch_geometric.loader.DataLoader
        train_loader = DataLoader(
                train_data,
                batch_size=self.train_params['batch_size'],
                shuffle=True,
                num_workers=0,
            )
        valid_loader = DataLoader(
                valid_data,
                batch_size=self.train_params['batch_size'],
                shuffle=False,
                num_workers=0,
            )
        
        return train_loader, valid_loader, train_data
        
    def get_classifier_loss_fn(self):
        alpha = torch.tensor([1/4]*self.train_params['outputs'])
        loss_fn = FocalLoss(alpha = alpha.float(), gamma = self.train_params['gamma'], reduction = 'mean')
        return loss_fn

    def load_sequence_datasets(self, chains, fold):
        dataset = self.train_params['dataset']
        df = pd.read_csv('msa_data/' + dataset + '_msa.csv')
        df = df.sample(frac=1,random_state=1)
        train_ind = (np.array(list(np.where(df['split'] == "train")[0]) + list(np.where(df['split'] == 'test')[0])),)
        num_folds = self.train_params['num_folds']
        train_size = len(train_ind[0])
        fold_size = train_size // num_folds
        
        if fold != (num_folds - 1):
            test_ind = (np.array(train_ind[0][fold*fold_size:(fold+1)*fold_size]),)
        else:
            test_ind = (np.array(train_ind[0][fold*fold_size:]),)
        train_ind = (np.array([x for x in train_ind[0] if x not in test_ind[0]]),)
        self.train_params['down_sample'] = False
        #tokenzings both fvs or single fv depending on dataset
        vocab_path = 'ablang/vocab.json'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = ABtokenizer(vocab_path)
        if chains =='both':
            heavy_tokens = tokenizer(df['heavy_msa'], pad=True, device = device)
            light_tokens = tokenizer(df['light_msa'], pad=True, device = device)
            X = torch.cat([heavy_tokens, light_tokens], dim =1)
        else:
            X = tokenizer(df['msa'], pad=True, device = device)
        
        #normalizing target if specified
        if self.train_params['normalize']:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(torch.tensor(df['target']).reshape(-1,1))
            Y = torch.tensor(scaled)
        else:
            Y = torch.tensor(df['target'])
        
        #Getting into dataset and dataloader format
        dataset = self.SequenceDataset(X,Y)
        
        train, test = Subset(dataset, train_ind), Subset(dataset, test_ind)
        batch_size =  self.train_params['batch_size']
        
        train_loader = DataLoader([[a,b] for a,b in zip(train[0][0], 
                                                       train[0][1])], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader([[a,b] for a,b in zip(test[0][0], 
                                                      test[0][1])], batch_size=batch_size, shuffle=False)    
        return train_loader, valid_loader, train
    
    def get_regression_loss_fn(self):
        if self.train_params['loss_fn'] == 'mse':
            return torch.nn.MSELoss()
        elif self.train_params['loss_fn'] == 'mae':
            return torch.nn.L1Loss()
        return None
    
    def get_accuracy_metric(self, pred, labels):
        if self.train_params['outputs'] > 1:
            correct = total = 0
            for outputs, true in zip(pred, labels):
                predicted  = torch.max(outputs,1)[1]
                total += true.size(0)
                correct += (predicted == true).sum().item()
            score = 100*correct/total
        else:
            predicted = torch.cat(pred)
            true = torch.cat(labels)
            score = spearmanr(predicted, true).correlation
        return score
    
    
    def save_results(self, score, fold):
        path = 'outputs/'
        outputs_path = path + (self.train_params['dataset'] + '/' + self.train_params['model_choice'] + '/')
        curr_id = str(max([int(file.split(".")[0][3:]) for file in os.listdir(outputs_path)]) + 1)
        
        print("Saving Results, MODEL ID: {}".format(curr_id))
        
        with open(outputs_path + 'out' + curr_id + '.txt','w') as fh:
            fh.write("Best Score: {:.2f}\n".format(score))
            fh.write("Fold {}".format(fold))
            fh.write("Training Params:\n")
            fh.write(str(self.train_params))

        return curr_id
        
    def save_model(self, score, model, fold):
        temp_path = 'temp/' + self.train_params['dataset'] + '/' + self.train_params['model_choice'] + '/'
        torch.save(model.state_dict(), temp_path + 'best_' + str(fold))
        print("Best Score for Fold! Model and Score Saved in Temp Folder")
    
    
    def get_warmup_scheduler(self, optimizer, train_loader):
        lr_scheduler = lrs.LinearLR(optimizer, start_factor=1.0, 
                                          end_factor=0.1, total_iters=30, 
                                          last_epoch=- 1, verbose=False)
        warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                       len_loader=len(train_loader),
                                       warmup_steps=10,
                                       warmup_start_lr=1e-8,
                                       warmup_mode='linear')
            
        return warmup_scheduler
            
    
    def train_and_test(self, fold):
        train_params = self.train_params
        model_choice = train_params['model_choice']
        print("FOLD NUMBER: {}".format(fold+1))
        print("Loading Datasets and Dataloader")
        if model_choice == 'linear':
            train_dataloader, valid_dataloader, train_dataset = self.load_sequence_datasets(train_params['chains'], fold)
        else:
            train_dataloader, valid_dataloader, train_dataset = self.load_structure_datasets(train_params['chains'], fold)

        print("Initializing Model")
        model = self.init_model(self.train_params, train_dataset, model_choice)
        
         
        print("Getting Loss Function and Optimizer")
        outputs = train_params['outputs']
        if outputs > 1:
            loss_fn = self.get_classifier_loss_fn()
        else:
            loss_fn = self.get_regression_loss_fn()
        
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.train_params['lr'],
                                              betas=self.train_params['adam_betas'], 
                                              eps= self.train_params['adam_epsilon'], 
                                              weight_decay= self.train_params['weight_decay'])
        

        if train_params['dynamic_lr']:
            warmup_scheduler = self.get_warmup_scheduler(optimizer,train_dataloader)
        
          
        print("Freezing Encoder")  
        fzn_lyrs = train_params['frozen_layers']
        model.train()
        
        if train_params['chains'] == 'both':
            self.freeze_encoder_l(model, fzn_lyrs)
            self.freeze_encoder_h(model, fzn_lyrs)
        else:
            self.freeze_encoder(model, fzn_lyrs)
        
        stopping_counter = 0
        score = best_score = -np.inf
        
        print("Beginning Training")
        for epoch in range(self.train_params['num_epochs']):  # loop over the dataset multiple times
            if stopping_counter >= self.train_params['stopping_allowance']:
                break
            running_loss = 0.0
            for i, batch in enumerate(train_dataloader, 0):
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                inputs, labels = batch
                outputs = model(inputs).cpu()
                if train_params['outputs'] < 2:
                    labels = labels.to(torch.float32)
                    outputs = torch.reshape(outputs,labels.shape)
                else:
                    labels = labels.to(torch.int64)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                if train_params['dynamic_lr']:
                    warmup_scheduler.step()
                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:    # print every 50 mini-batches
                    print('[%d, %5d] Training Loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0
            
            
            #Getting Validation Score and Saving if score is higher than threshold
            model.eval()
            predicted = []
            true = []
            with torch.no_grad():
                for i, batch in enumerate(valid_dataloader, 0):
                    inputs, labels = batch
                    predicted.append(model(inputs).cpu())
                    true.append(labels)
            score = self.get_accuracy_metric(predicted,true)
            print("Valid Score: {:.5f}".format(score))
           
            #Early Stopping
            delta = score - best_score          
            if delta > 0:
                best_score = score
                stopping_counter = 0
                self.save_model(score, model, fold)  
                print("Stopping Counter Reset, Saving Model in Temp")
            else:
                stopping_counter +=1
                print("Stopping Counter: {}".format(stopping_counter))
        
        #Saving Results
        self.save_results(best_score, fold)
 
        return best_score
    
if __name__ == "__main__":
    
    train_params = {'activation': 'relu', 'adam_betas': (0.9, 0.97), 
                    'adam_epsilon': 1e-08, 'batch_size': 16, 'cdr_mask': False, 
                    'conv_layers': 4, 'down_sample': True, 'drop_rate': 0.4, 
                    'dynamic_lr': True, 'freeze': True, 'frozen_layers': 9, 
                    'gamma': 2.0, 'int_size': 768, 'layer_norm_epsilon': 1e-12, 
                    'loss_fn': 'focal', 'lr': 4.777344017623903e-05, 'n_hidden': 1.5, 
                    'no_decay': True, 'normalize': False, 'num_epochs': 100, 
                    'stopping_allowance': 5, 'top_k': 20, 'universal_pooling': False, 
                    'weight_decay': 0.01, 'weights': False, 'outputs': 4, 'dataset': 'mers', 
                    'model_choice': 'gvp', 'chains': 'heavy', 'pretrained': True, 'num_folds':5}
    print(train_params)
    trainer = CVTrainAbLangGNN(train_params)
    score = trainer.train_and_test()    
    print("Final Score: {}".format(score))


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
