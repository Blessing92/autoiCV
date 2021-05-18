'''
Build NARM model
'''

from __future__ import print_function
from collections import OrderedDict
import gc
import os
import sys
import nni
import time
import theano
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from theano import config
import theano.tensor as T
from data_process import prepare_data
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# Set the random number generators' seeds for consistency
SEED = 42
np.random.seed(SEED)

logger = logging.getLogger('NARM_AutoML')

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def dropout_layer(state_before, use_noise, trng, drop_p=0.5):
    retain = 1. - drop_p
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape,
                                                             p=retain, n=1,
                                                             dtype=state_before.dtype)), state_before * retain)
    return proj

def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_params(options):
    """
    Global (not GRU) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    params['Wemb'] = init_weights((options['n_items'], int(options['dim_proj'])))
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # attention
    params['W_encoder'] = init_weights((int(options['hidden_units']), int(options['hidden_units'])))
    params['W_decoder'] = init_weights((int(options['hidden_units']), int(options['hidden_units'])))
    params['bl_vector'] = init_weights((1, int(options['hidden_units'])))
    params['bili'] = init_weights((int(options['dim_proj']), 2 * int(options['hidden_units'])))
    return params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def init_weights(shape):
    sigma = np.sqrt(2. / shape[0])
    return numpy_floatX(np.random.randn(*shape) * sigma)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_gru(options, params, prefix='gru'):
    """
    Init the GRU parameter:

    :see: init_params
    """
    Wxrz = np.concatenate([init_weights((int(options['dim_proj']), int(options['hidden_units']))),
                           init_weights((int(options['dim_proj']), int(options['hidden_units']))),
                           init_weights((int(options['dim_proj']), int(options['hidden_units'])))], axis=1)
    params[_p(prefix, 'Wxrz')] = Wxrz

    Urz = np.concatenate([ortho_weight(int(options['hidden_units'])),
                          ortho_weight(int(options['hidden_units']))], axis=1)
    params[_p(prefix, 'Urz')] = Urz

    Uh = ortho_weight(int(options['hidden_units']))
    params[_p(prefix, 'Uh')] = Uh

    b = np.zeros((3 * int(options['hidden_units']),))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_):
        preact = T.dot(h_, tparams[_p(prefix, 'Urz')])
        preact += x_[:, 0:2 * int(options['hidden_units'])]

        z = T.nnet.hard_sigmoid(_slice(preact, 0, int(options['hidden_units'])))
        r = T.nnet.hard_sigmoid(_slice(preact, 1, int(options['hidden_units'])))
        h = T.tanh(T.dot((h_ * r), tparams[_p(prefix, 'Uh')]) + _slice(x_, 2, int(options['hidden_units'])))

        h = (1.0 - z) * h_ + z * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    state_below = (T.dot(state_below, tparams[_p(prefix, 'Wxrz')]) +
                   tparams[_p(prefix, 'b')])

    hidden_units = int(options['hidden_units'])
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=T.alloc(numpy_floatX(0.), n_samples, hidden_units),
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval

layers = {'gru': (param_init_gru, gru_layer)}


def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """

    updates = OrderedDict()
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g  # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2   # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)               # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)               # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates[m_previous] = m
        updates[v_previous] = v
        updates[theta_previous] = theta
    updates[t] = t + 1.

    return updates


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                int(options['dim_proj'])])
    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng, drop_p=0.25)

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    def compute_alpha(state1, state2):
        tmp = T.nnet.hard_sigmoid(T.dot(tparams['W_encoder'], state1.T) + T.dot(tparams['W_decoder'], state2.T))
        alpha = T.dot(tparams['bl_vector'], tmp)
        res = T.sum(alpha, axis=0)
        return res

    last_h = proj[-1]

    sim_matrix, _ = theano.scan(
        fn=compute_alpha,
        sequences=proj,
        non_sequences=proj[-1]
    )
    att = T.nnet.softmax(sim_matrix.T * mask.T) * mask.T
    p = att.sum(axis=1)[:, None]
    weight = att / p
    atttention_proj = (proj * weight.T[:, :, None]).sum(axis=0)

    proj = T.concatenate([atttention_proj, last_h], axis=1)

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng, drop_p=0.5)

    ytem = T.dot(tparams['Wemb'], tparams['bili'])
    pred = T.nnet.softmax(T.dot(proj, ytem.T))
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, cost


def pred_evaluation(f_pred_prob, prepare_data, data, iterator, item_freqs, k = 20):
    """
    Compute recall@20 and mrr@20
    f_pred_prob: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    recall = 0.0
    mrr = 0.0
    evalutation_point_count = 0
    preds_freqs = []
    preds_items = []
    #print('HEEELLLOOO', len(item_freqs))
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        preds = f_pred_prob(x, mask)
        #print(preds.shape)
        targets = y
        ranks = (preds.T > np.diag(preds.T[targets])).sum(axis=0) + 1
        rank_ok = (ranks <= k)
        recall += rank_ok.sum()
        mrr += (1.0 / ranks[rank_ok]).sum()
        evalutation_point_count += len(ranks)
        for i in range(preds.shape[0]):
            series = pd.Series(data = preds[i])
            s = series.nlargest(k).index.values
            for r in s:
                preds_items.append(r)
                preds_freqs.append(item_freqs[r])

    recall = numpy_floatX(recall) / evalutation_point_count
    mrr = numpy_floatX(mrr) / evalutation_point_count
    eval_score = (recall, mrr, 
                  len(list(set(preds_items))) / len(item_freqs.keys()), 
                  (np.mean(preds_freqs) / max(item_freqs.values())) )
    return eval_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--encoder', type=str, default='gru')
    parser.add_argument('--use_dropout', type=bool, default=True, help='Dropout')
    parser.add_argument('--dim_proj', type=int, default=100, help='Item embedding dimension. initial:50')
    parser.add_argument('--hidden_units', type=int, default=100, help='Number of GRU hidden units. initial:100')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--itemid', default='ItemID', type=str)
    parser.add_argument('--sessionid', default='SessionID', type=str)
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
        data[['tmp']] = data[[itemid]].apply(lambda x: x.cat.codes + 1)
        freqs = dict(data['tmp'].value_counts())
    #$$$%^&*()(*&^%$ FREQ PART
    cold_items = []
    if Train == False and freq != 0:
        for i, row in data.iterrows():
            if freq >= row['freq_threshold']:
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
                if cnt_pattern[-1] in cold_items or freq == 0:
                    patterns.append(cnt_pattern[:-1])
                    labels.append(cnt_pattern[-1])
    
    return (patterns, labels), itemsIDs, freqs, old_new

def train_gru(args):
    #patience=100, saveto='gru_model.npz',
    #is_valid=True, is_save=False, reload_model=None, test_size=-1
    exps = pd.read_csv('exp.csv')
    for i,row in exps.iterrows():
        gc.collect()
        args['expname'] = row['name']
        args['sessionid'] = row['SessionID']
        args['itemid'] = row['ItemID']
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
        freqs[0] = 0
        print('Start Data Preprocessing: Testing Set')
        valid, _, _, _ = load_sequence(args['data_folder'] + '/' + args['valid_data'], 
                                       args['itemid'], args['sessionid'], Train = False, 
                                       itemsIDs = itemsIDs, freq = args['freq'], 
                                       old_new = old_new)
        print("%d train examples." % len(train[0]))
        print("%d valid examples." % len(valid[0]))
        # Model options
        params = init_params(args)
        tparams = init_tparams(params)
        (use_noise, x, mask, y, f_pred_prob, cost) = build_model(tparams, args)
        all_params = list(tparams.values())
        updates = adam(cost, all_params, args['lr'])
        train_function = theano.function(inputs=[x, mask, y], outputs=cost, updates=updates)
        uidx = 0  # the number of update done
        cPid = os.getpid()
        command_memory ="python memoryLogger.py "+ str(cPid) + " " + args['expname'] + "train"
        #memory_task = subprocess.Popen(command_memory, stdout=subprocess.PIPE, shell=True)
        try:
            t1 = time.time()
            for eidx in range(args['epoch']):
                n_samples = 0
                epoch_loss = []
                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(train[0]), int(args['batch_size']), shuffle=True)
                for _, train_index in kf:
                    uidx += 1
                    use_noise.set_value(1.)
                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index]
                    x = [train[0][t]for t in train_index]
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]
                    loss = train_function(x, mask, y)
                    epoch_loss.append(loss)
                    if np.isnan(loss) or np.isinf(loss):
                        print('bad loss detected: ', loss)
                        return 1., 1., 1.
                print('Epoch ', eidx, 'Loss ', np.mean(epoch_loss))
                
                # Report intermediate result to the tuner
                nni.report_intermediate_result(np.mean(epoch_loss))
                logger.debug('test loss %g', np.mean(epoch_loss))
                logger.debug('Pipe send intermediate result done')
                
                use_noise.set_value(0.)
                if eidx % 3 == 0:
                    kf_valid = get_minibatches_idx(len(valid[0]), int(args['batch_size']))
                    valid_eval = pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid, freqs)
                    print('Valid Recall@20:', valid_eval[0], '\nValid Mrr@20:', valid_eval[1])
                    
            # Report intermediate result to the tuner
            nni.report_final_result(np.mean(epoch_loss))
            logger.debug('Final loss is %g', np.mean(epoch_loss))
            logger.debug('Send final result done')
            
        except KeyboardInterrupt:
            print("Training interupted")
        #memory_task.kill()
        train_time = time.time() - t1
        use_noise.set_value(0.)
        t1 = time.time()
        Ks = [1, 3, 5, 10, 30]
        hit = [0, 0, 0, 0, 0]; MRR = [0 ,0, 0, 0, 0]
        cov = [0, 0, 0, 0, 0]; pop = [0, 0, 0, 0, 0];
        command_memory ="python memoryLogger.py "+ str(cPid) + " " + args['expname'] + "test"
        #memory_task = subprocess.Popen(command_memory, stdout=subprocess.PIPE, shell=True)
        for k in range(len(Ks)):
            kf_valid = get_minibatches_idx(len(valid[0]), int(args['batch_size']))
            results = pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid, freqs, Ks[k])
            hit[k] = results[0]
            MRR[k] = results[1]
            cov[k] = results[2]
            pop[k] = results[3]
        test_time = time.time() - t1
        #memory_task.kill()
        print('==================================================')
        print('Recall:', hit,'\nMRR:', MRR, '\nCoverage:', cov, '\nPopularity:', pop)
        print ('\ntrain_time:', train_time, '\nTest time:', test_time / len(Ks))
        print('End Model Predictions')
        
        # Print experiment to the logger
        print('===================================================')
        print("LOGGER_" + args['expname'])
        print(str(hit[0])+','+str(hit[1])+','+str(hit[2])+','+str(hit[3])+','+str(hit[4])+','+str(MRR[0])+','+str(MRR[1])+','+str(MRR[2])+','+str(MRR[3])+','+str(MRR[4]))
        print("\nCOV:"+str(cov[0])+','+str(cov[1])+','+str(cov[2])+','+str(cov[3])+','+str(cov[4]))
        print("\nPOP:"+str(pop[0])+','+str(pop[1])+','+str(pop[2])+','+str(pop[3])+','+str(pop[4]))
        print("\nTrainTime:"+str(train_time))
        print("\nTestTime:"+str(test_time))
        
        with open("LOGGER_"+ args['expname'] + ".txt", "a") as myfile:
            myfile.write(str(hit[0])+','+str(hit[1])+','+str(hit[2])+','+str(hit[3])+','+str(hit[4])+','+str(MRR[0])+','+str(MRR[1])+','+str(MRR[2])+','+str(MRR[3])+','+str(MRR[4]))
            myfile.write("\nCOV:"+str(cov[0])+','+str(cov[1])+','+str(cov[2])+','+str(cov[3])+','+str(cov[4]))
            myfile.write("\nPOP:"+str(pop[0])+','+str(pop[1])+','+str(pop[2])+','+str(pop[3])+','+str(pop[4]))
            myfile.write("\nTrainTime:"+str(train_time))
            myfile.write("\nTestTime:"+str(test_time))
            myfile.write("\n############################################\n")
if __name__ == '__main__':
    try:
        # Get parameters from tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        tuner_params['epoch'] = tuner_params['TRIAL_BUDGET'] * 100
        params = vars(parse_args())
        params.update(tuner_params)
        train_gru(params)
    except Exception as exception:
        logger.exception(exception)
        raise                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ()
