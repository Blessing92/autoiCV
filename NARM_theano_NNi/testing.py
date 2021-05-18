import argparse
from collections import OrderedDict

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
    parser.add_argument('--train_data', default='4-4(64 portion).csv', type=str)
    parser.add_argument('--data_folder', default='~/Documents/Datasets/RecSys', type=str)
    
    args, _ = parser.parse_known_args()
    return args

argument= {'epoch': 50, 'batch_size': 128, 'encoder': 'gru', 'use_dropout': True, 'dim_proj': 50, 
            'hidden_units': 50, 'lr': 0.003, 'itemid': 'ItemID', 'sessionid': 'SessionID', 
            'valid_data': 'recSys15Valid.csv', 'train_data': '4-4(64 portion).csv', 
            'data_folder': '~/Documents/Datasets/RecSys', 'TRIAL_BUDGET': 1, 'expname': 'recsys', 
            'freq': 0, 'n_items': 21988}

print("VARS(ARGS)====> ", vars(argument))

def init_params():
    """
    Global (not GRU) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    params['Wemb'] = 0.2
#    params = get_layer(options['encoder'])[0](options,
#                                              params,
#                                              prefix=options['encoder'])
    # attention
    params['W_encoder'] = 5
    params['W_decoder'] = 2
    params['bl_vector'] = 7
    params['bili'] = 0.22
    return params

print(vars(init_params()))
print("#######################################")
print(parse_args())
print("========================================")
print(vars(parse_args()))