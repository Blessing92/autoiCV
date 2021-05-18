 # coding=utf-8
import os
import time
import nni
import argparse
import logging
import numpy as np
#import tensorflow as tf
from util.Randomer import Randomer
from model.STAMP_rsc import Seq2SeqAttNN
from data_prepare.data_read_p import load_data_p

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

logger = logging.getLogger('STAMP_AutoML')



#Embeddings Initialization
def load_random(word2idx, edim = 300, init_std = 0.05):
    emb_dict = np.random.normal(0, init_std, [len(word2idx), edim]) #items embeddings dimension edim are initialized with normal distribution with standard deviation = init_std
    emb_dict[0] = [0.0] * edim #padding embedding vector is initialized with zeros
    return emb_dict

#Loading Train and Test Data Splits
def load_tt_datas(args, config={}):
    '''
    load data.
    config: pre_embedding.
    '''
    train_data = os.path.join(args['data_folder'], args['train_data'])
    valid_data = os.path.join(args['data_folder'], args['valid_data'])
    print( "Loading the Dataset.....")
    train_data, test_data, item2idx, config["n_items"] = load_data_p(train_data, valid_data, args, pro = args['sample'])
    config['pre_embedding'] = load_random(item2idx, edim=config['hidden_size'], init_std=config['emb_stddev'])
    print("Finished Loading the Dataset\n---------------")
    return train_data, test_data


def load_conf(args):
    param_conf = {'cell' : 'gru'} #Recurrent Unit
    param_conf['model_save_path'] = 'ckpt/' #Checkpoints Save Path
    param_conf['stddev'] = args['stddev'] #standard Deviation for Weight Initialization
    param_conf['emb_stddev'] = args['stddev'] #standard Deviation for Embedding Initialization
    param_conf['update_lr'] = False #LR decay
    param_conf['emb_up'] = True #Use embeddings
    param_conf['is_print'] = True #Print intermediate results
    param_conf['hidden_size'] = args['hidden'] #hidden Size
    param_conf['edim'] = args['edim'] #embedding dimensions
    param_conf['init_lr'] = args['lr'] #initial LR
    param_conf['nepoch'] = args['epoch'] #number of epochs
    param_conf['batch_size'] = args['batch'] #batch size
    param_conf['max_grad_norm'] = args['max_grad_norm'] #maximum norm of gradient
    param_conf['cut_off'] = args['K'] #Recall@K and MRR@K
    param_conf['active'] = args['activation'] #activation function
    return param_conf

def main(args):
    is_train = not args['nottrain']
    is_save = args['savemodel']
    model_path = args['modelpath']
    epoch = args['epoch']
    config = load_conf(args)
    config['nepoch'] = epoch
    #Load Dataset
    train_data, test_data = load_tt_datas(args, config)
    # setup randomer
    Randomer.set_stddev(config['stddev'])
    with tf.Graph().as_default():
        # build model
        model = Seq2SeqAttNN(config)
        model.build_model()
        if is_save or not is_train:
            saver = tf.train.Saver(max_to_keep=30)
        else:
            saver = None
        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if is_train:
                t1 = time.time()
                model.train(sess, train_data, test_data, saver)
                t2 = time.time()
                print('Training+Evaluation Time = ', t2 - t1)
            else:
                t1 = time.time()
                saver.restore(sess, model_path)
                model.test(sess, test_data)
                t2 = time.time()
                print('Testing Time = ', t2 - t1)

def get_params():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
    parser.add_argument('--max_grad_norm', type=int, default=150, help="Maximum Gradient Norm")
    parser.add_argument('--edim', type=int, default=100, help="Embedding Dimensions")
    parser.add_argument('--sample', type=int, default=512, help="Use last 1/sample portion of the dataset")
    parser.add_argument('--hidden', type=int, default=100, help="Network Hidden Size")
    parser.add_argument('--batch', type=int, default=512, help="Batch size for the training process")
    parser.add_argument('--activation', type=str, default='sigmoid', help="Activation function (relu, sigmoid, tanh)")
    parser.add_argument('--lr', type=float, default=0.003, help="Learning Rate")
    parser.add_argument('--stddev', type=float, default=0.05, help="Standard Deviation")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--modelpath", type=str, default='') #model path in case of doing testing
    parser.add_argument("--savemodel", default=True) #save model or not
    parser.add_argument("--nottrain", default=False) #Set true if using pretrained model for testing
    parser.add_argument('--itemid', default='ItemID', type=str)
    parser.add_argument('--sessionid', default='SessionID', type=str)
    parser.add_argument('--valid_data', default='recSys15Valid.csv', type=str)
    parser.add_argument('--train_data', default='4-4(64 portion).csv', type=str)
    parser.add_argument('--data_folder', default='~/Documents/Datasets/RecSys', type=str)
    
    args, _ = parser.parse_known_args()
    
    print('Finished Reading Data \nStart Model Fitting...')
    
    return args


if __name__ == '__main__': 
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        tuner_params['epoch'] = tuner_params['TRIAL_BUDGET'] * 10
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
