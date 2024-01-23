from glob import glob
from collections import Counter, OrderedDict, defaultdict
import joblib
import torch
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll import scope
from cv_train_ablang_gnn import CVTrainAbLangGNN
import argparse
import numpy as np
import subprocess

#setting up argparse to get model we wish to train and parameters we wish to vary, seperate them by commas
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vary", help="Hyperparameter to vary")
parser.add_argument("-he","--head", help="Head to use")
parser.add_argument("-o", "--outputs", help="Number of Nodes in Final Output")
parser.add_argument("-d", "--dataset", help="Dataset to train on")
parser.add_argument("-c", "--chain", help="heavy,light, both")
parser.add_argument("-n", "--n_trials", help="number of trials for optimization")
parser.add_argument("-p", "--pretrained", help="y for pretrained n for naive graph head")
parser.add_argument("-k", "--k_folds", help="number of folds for cross validation")
parser.add_argument("-t", "--titration", help="list of sample percentages separated with commas")

#getting model and parameters we wish to vary or setting defaults if left unspecified
args = parser.parse_args()
head = args.head if args.head != None else 'linear'
common_hps = ['lr','batch_size','n_hidden','frozen_layers','drop_rate','weight_decay','adam_betas','dynamic_lr']
common_hps = common_hps + ['conv_layers','top_k'] if head == 'gat' else common_hps
to_vary = args.vary.split(',') if args.vary != None else common_hps
dataset = args.dataset
outputs = 2 if dataset == 'agg' else 1
chains_dict = {'mers':'heavy','mrgx':'heavy','agg':'light'}
chains = chains_dict[dataset] if dataset in chains_dict.keys() else 'both'
n_trials = int(args.n_trials) if args.n_trials != None else 500
k_folds = int(args.k_folds) if args.k_folds != None else 5
titration = [float(x) for x in args.titration.split(",")] if args.titration != None else False
pretrained = (args.pretrained == 'y') if args.pretrained != None else True

#default parameters

static_space = {
        "activation" : hp.choice("activation", ['relu']),
        "adam_betas": hp.choice("adam_betas", [[0.9,0.999]]),
        "adam_epsilon": hp.choice("adam_epsilon",[1e-8]),
        "layer_norm_epsilon": hp.choice("layer_norm_epsilon",[1e-12]),
        "weight_decay":hp.choice("weight decay", [0.01]),
        "batch_size": hp.choice("batch_size", [32]),
        "num_epochs": hp.choice("num_epochs", [50]),
        "stopping_allowance": hp.choice("stopping_allowance", [6]),
        "loss_fn":hp.choice("loss_fn",['mae']),
        "gamma":hp.choice("gamma",[2.0]),
        "weights":hp.choice("weights",[False]),
        "freeze":hp.choice("freeze",[True]),
        "lr":hp.choice("lr", [1e-5]),
        "no_decay":hp.choice("no_decay",[True]),
        "drop_rate":hp.choice('drop_rate',[0.2]),
        "frozen_layers":hp.choice('frozen_layers',[13]),
        "n_hidden":hp.choice('n_hidden',[2]),
        "conv_layers":hp.choice('conv_layers',[3]),
        'cdr_mask':hp.choice('cdr_mask',[False]),
        'top_k':hp.choice('top_k',[20]),
        'int_size':hp.choice('int_size',[768]),
        'dynamic_lr':hp.choice('dynamic_lr',[False]),
        'universal_pooling':hp.choice('universal_pooling',[False]),
        'normalize':hp.choice('normalize',[False]),
        'down_sample':hp.choice('down_sample',[False])
        }

#variable space of the hyperparameters
vary_space = {
            "activation" : hp.choice("activation", ['relu','tanh']),
            "adam_betas": hp.choice("adam_betas", [[0.9,0.98],[0.9,0.999],[0.9,0.99]]),
            "adam_epsilon": hp.choice("adam_epsilon",[1e-6,1e-7,1e-8,1e-9]),
            "weight_decay":hp.choice("weight decay", [0.01, 0.005, 0.02]),
            "batch_size": hp.choice("batch_size", [2,4,8,16]),
            "gamma":hp.choice("gamma",[1.0,2.0,3.0]),
            "loss_fn":hp.choice("loss_fn",['mae','mse']),
            "no_decay":hp.choice("no_decay",[True, False]),
            "lr":hp.uniform("learning_rate",1e-7,1e-4),
            "drop_rate":hp.choice('drop_rate',[0.2,0.3,0.4,0.5]),
            "frozen_layers":hp.choice('frozen_layers',[1,2,3,4,5,6,7,8,9,10,11]),
            "n_hidden":hp.choice('n_hidden',[0.25,0.5,1,1.5,2]),
            "conv_layers":hp.choice('conv_layers',[1,2,3,4]),
            'cdr_mask':hp.choice('cdr_mask',[True,False]),
            'top_k':hp.choice('top_k',[10,15,20,25,30]),
            'int_size':hp.choice('int_size',[384,768,1536]),
            'dynamic_lr':hp.choice('dynamic_lr',[True,False]),
            'universal_pooling':hp.choice('universal_pooling',[True, False]),
            'normalize':hp.choice('normalize',[True,False])
             }

vary_space['top_k'] = hp.choice('top_k',[20]) if head in ['mifst','gvp','linear'] else vary_space['top_k']
space['top_k'] = hp.choice('top_k',[20]) if head in ['mifst','gvp','linear'] else static_space['top_k']


#if we pass all, we vary all of them, else just the specified one
for key in to_vary:
    space[key] = vary_space[key]

print("VARYING {}".format(to_vary))

def get_global_best(best_score_and_param_path, dataset, head):
    with open(best_score_and_param_path + dataset + '_' + head, 'r') as fh:
        lines = fh.readlines()
        top_avg_score = float(lines[0].split()[1])
    return top_avg_score

def save_model_score_and_params(score, params, best_score_and_param_path):
    with open(best_score_and_param_path + params['dataset'] + '_' + params['model_choice'] , 'w') as fh:
        fh.write('Score: ' + str(score) + '\n')
        fh.write(str(params))
        print("CV Ensemble Run Params Saved")

def cross_validate(params):    
    print(params)
    params['outputs'] = int(outputs)
    params['dataset'] = dataset
    params['model_choice'] = head
    params['chains'] = chains
    params['pretrained'] = pretrained
    params['num_folds'] = k_folds
    trainer = CVTrainAbLangGNN(params)
    
    #paths where we save models and results
    temp_path = 'temp/' + dataset + '/' + head + '/' 
    best_global_model_path = 'outputs/best_models/' + dataset + '_' + head + '/'
    best_score_and_param_path = 'outputs/best_models/params/'
    
    scores = []
    #iterating over each k-fold
    for i in range(k_folds):
        
        #train_and_test will save the weights and score of the best running fold in the temp dir
        score = trainer.train_and_test(i)
        scores.append(score)
    
    avg_score = sum(scores)/k_folds
    st_err = 1.96*np.std(scores)/np.sqrt(k_folds)
    
    print("Average Score: {} +- {}".format(avg_score, st_err))
    
    #if the average across all folds is better than global average, we save the best fold as the global best
    global_best_avg = get_global_best(best_score_and_param_path,dataset,head)
    
    if avg_score > global_best_avg:
        print("Average Accuracy Beat Top Score! Saving All Folds for Ensembling and Run Params")
        for i in range(k_folds):
            subprocess.Popen(['mv', temp_path + 'best_' + str(i), best_global_model_path + 'fold' + str(i) + '.pth'])
            save_model_score_and_params(avg_score, params, best_score_and_param_path)
    
    return -avg_score



trials = Trials()
best = fmin(
    fn=cross_validate,
    space = space, 
    algo=tpe.suggest, 
    max_evals = n_trials, 
    trials=trials
)

scores = np.array([-x for x in trials.losses()])
avg_score = sum(scores)/n_trials
st_err = 1.96*np.std(scores)/np.sqrt(n_trials)
print("Average Score: {} +- {}".format(avg_score, st_err))
print("Best: {}".format(best))


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
