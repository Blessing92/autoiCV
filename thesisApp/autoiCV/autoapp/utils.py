import argparse


def spop_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5, help="K items to be used in Recall@K and MRR@K")
    parser.add_argument('--topn', type=int, default=100, help="Number of top items to return non zero scores for them (most popular)")
    parser.add_argument('--itemid', default='ItemID', type=str)
    parser.add_argument('--sessionid', default='SessionID', type=str)
    parser.add_argument('--valid_data', default='recSys15Valid.csv', type=str)
    parser.add_argument('--train_data', default='4-4(64 portion).csv', type=str)
    parser.add_argument('--data_folder', default='/Users/mungaperseverance/Downloads/RecSys15 Dataset Splits/RecSys', type=str)

    args = parser.parse_args(args=[])
    return args
