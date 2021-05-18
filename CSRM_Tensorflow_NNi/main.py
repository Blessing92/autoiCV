# coding:utf-8
from __future__ import absolute_import
#import tensorflow as tf
import gc
import os
import nni
from csrm import CSRM
import argparse
import logging
#import data_process
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

logger = logging.getLogger('CSRM_AutoML')

def parse_args():
    parser = argparse.ArgumentParser(description="Run CSRM.")
    parser.add_argument('--dataset', nargs='?', default='lastfm', help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=25, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--n_items', type=int, default=39164, help='Item size 37484, 39164')
    parser.add_argument('--dim_proj', type=int, default=100, help='Item embedding dimension. initial:50')
    parser.add_argument('--hidden_units', type=int, default=100, help='Number of GRU hidden units. initial:100')
    parser.add_argument('--display_frequency', type=int, default=5, help='Display to stdout the training progress every N updates.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--keep_probability', nargs='?', default='[0.75,0.5]', help='Keep probability (i.e., 1-dropout_ratio). 1: no dropout.')
    parser.add_argument('--no_dropout', nargs='?', default='[1.0,1.0]', help='Keep probability (i.e., 1-dropout_ratio). 1: no dropout.')
    parser.add_argument('--memory_size', type=int, default=512, help='.')
    parser.add_argument('--memory_dim', type=int, default=100, help='.')
    parser.add_argument('--shift_range', type=int, default=1, help='.')
    parser.add_argument('--controller_layer_numbers', type=int, default=0, help='.')
    parser.add_argument('--itemid', default='itemid', type=str)
    parser.add_argument('--sessionid', default='sessionid', type=str)
    parser.add_argument('--valid_data', default='recSys15Valid.csv', type=str)
    parser.add_argument('--train_data', default='4-3(16 portion).csv', type=str)
    parser.add_argument('--data_folder', default='~/Documents/Datasets/RecSys', type=str)
    
    args, _ = parser.parse_known_args()
    
    return args


def load_sequence(from_path, itemid, sessionid, Train = True, itemsIDs = [], freq = 0, old_new = {}):
    freqs = {}
    data = pd.read_csv(from_path)
    if Train == True:
        itemsIDs = list(data[itemid].unique())
        data[itemid] = data[itemid].astype('category')
        new_old = dict(enumerate(data[itemid].cat.categories))
        old_new = {y:x for x,y in new_old.items()}
        data[['tmp']] = data[[itemid]].apply(lambda x: x.cat.codes+1)
        freqs = dict(data['tmp'].value_counts())
    #$$$%^&*()(*&^%$ FREQ PART
    cold_items = []
    if Train == False and freq != 0:
        for i, row in data.iterrows():
            if freq >= row['freq_threshold'] or freq == 0:
                cold_items.append(old_new[row[itemid]])
    #$$$%^&*()(*&^%$ FREQ PART
        
    patterns = []
    labels = []
    cnt_session = -1
    cnt_pattern = []
    for i in range(len(data)):
        if i % 100000 == 0:
            print('Finished Till Now: ', i)
        sid = data.loc[i, [sessionid]][0]
        iid = data.loc[i, [itemid]][0]
        if sid != cnt_session:
            cnt_session = sid
            cnt_pattern = []
        if Train == False and iid not in itemsIDs:
            continue
        cnt_pattern.append(old_new[iid]+1 )        
        if len(cnt_pattern) > 1:
            lst_pattern = []
            if len(patterns) > 0:
                lst_pattern = patterns[-1]
            if cnt_pattern != lst_pattern:
                if len(cold_items) == 0 or cnt_pattern[-1] in cold_items:
                    patterns.append(cnt_pattern[:-1])
                    labels.append(cnt_pattern[-1])
    return (patterns, labels), itemsIDs, freqs, old_new

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #print ("args: ", args)
    exps = pd.read_csv('exp.csv')
    for i,row in exps.iterrows():
        gc.collect()
        args['expname'] = row['name']
        args['sessionid'] = row['sessionid']
        args['itemid'] = row['itemid']
        args['data_folder'] = row['path']
        args['valid_data'] = row['test']
        args['train_data'] = row['train']
        args['freq'] = row['freq']
        
        print('Train:', args['train_data'], ' -- Test:', args['valid_data'], ' -- Freq:', args['freq'])
        with open("LOGGER_"+ args['expname'] + ".txt", "a") as myfile:
            myfile.write(row['train'] + ", " + row['test'] +"\n")
        
        # split patterns to train_patterns and test_patterns
        print('Start Data Preprocessing: Training Set')
        train, itemsIDs, freqs, old_new = load_sequence(args['data_folder'] + '/' + args['train_data'], 
                                                        args['itemid'], args['sessionid'], 
                                                        itemsIDs = [])
        args['n_items'] = len(itemsIDs) + 1
        print('Start Data Preprocessing: Testing Set')
        valid, _, _, _ = load_sequence(args['data_folder'] + '/' + args['valid_data'], 
                                       args['itemid'], args['sessionid'], Train = False, 
                                       itemsIDs = itemsIDs, freq = args['freq'], 
                                       old_new = old_new)
    
        #train, valid, test = data_process.load_data()
        print("%d train examples." % len(train[0]))
        print("%d valid examples." % len(valid[0]))
        keep_probability = np.array(args['keep_probability'])
        no_dropout = np.array(args['no_dropout'])
        result_path = "./save/" + args['dataset']
        # Build model
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            model = CSRM(sess=sess, n_items=args['n_items'], dim_proj=int(args['dim_proj']),
                hidden_units=int(args['hidden_units']), memory_size=args['memory_size'],
                memory_dim=args['memory_dim'], shift_range=args['shift_range'], lr=args['lr'],
                controller_layer_numbers=args['controller_layer_numbers'], batch_size=args['batch_size'],
                epoch=args['epoch'], keep_probability=keep_probability, no_dropout=no_dropout,
                display_frequency=args['display_frequency'], item_freqs = freqs, 
                expname = args['expname'])
            hit, MRR, cov, pop, train_time, test_time = model.train(train, valid, valid, result_path)
        
        print("#########################################################")
        print("NEW_LOGGER_ " + args['expname'])
        print(str(hit[0])+','+str(hit[1])+','+str(hit[2])+','+str(hit[3])+','+str(hit[4])+','+str(MRR[0])+','+str(MRR[1])+','+str(MRR[2])+','+str(MRR[3])+','+str(MRR[4]))
        print("\nCOV:"+str(cov[0])+','+str(cov[1])+','+str(cov[2])+','+str(cov[3])+','+str(cov[4]))
        print("\nPOP:"+str(pop[0])+','+str(pop[1])+','+str(pop[2])+','+str(pop[3])+','+str(pop[4]))
        print("\nTrainTime:"+str(train_time))
        print("\nTestTime:"+str(test_time))
        
        with open("NEW_LOGGER_"+ args['expname'] + ".txt", "a") as myfile:
            myfile.write(str(hit[0])+','+str(hit[1])+','+str(hit[2])+','+str(hit[3])+','+str(hit[4])+','+str(MRR[0])+','+str(MRR[1])+','+str(MRR[2])+','+str(MRR[3])+','+str(MRR[4]))
            myfile.write("\nCOV:"+str(cov[0])+','+str(cov[1])+','+str(cov[2])+','+str(cov[3])+','+str(cov[4]))
            myfile.write("\nPOP:"+str(pop[0])+','+str(pop[1])+','+str(pop[2])+','+str(pop[3])+','+str(pop[4]))
            myfile.write("\nTrainTime:"+str(train_time))
            myfile.write("\nTestTime:"+str(test_time))
            myfile.write("\n############################################\n")


if __name__ == '__main__': 
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        tuner_params['epoch'] = tuner_params['TRIAL_BUDGET'] * 100
        params = vars(parse_args())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise