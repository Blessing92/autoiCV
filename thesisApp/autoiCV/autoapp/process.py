import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def predict(features_df):
    # load the model from disk
    loaded_model = pickle.load(open('autoapp/model_picle', 'rb'))
    result = loaded_model.predict(features_df)

    if result[0] == 10:
        return 'STAMP'
    elif result[0] == 3:
        return 'S-POP'
    elif result[0] == 8:
        return 'CSRM'
    elif result[0] == 0:
        return 'NARM'
    elif result[0] == 9:
        return 'NextItNet'
    elif result[0] == 11:
        return 'SRGNN'
    elif result[0] == 7:
        return 'VSKNN'
    else:
        return 'Need more processing time'


def extract_dataset_properties(file):
    top_drop = []
    properties = {}
    data = pd.read_csv(file, sep=',')
    # remove space for each column
    data.columns = data.columns.str.strip()

    data.sort_values(['SessionID', 'ItemID'], inplace=True)
    data = data.reset_index(drop=True)
    for i, row in data.iterrows():
        if i == 0:
            old_row = row
            continue
        if old_row['SessionID'] == row['SessionID'] and old_row['ItemID'] == row['ItemID']:
            top_drop.append(i)
        old_row = row

    data.drop(top_drop, axis=0, inplace=True)

    # Total number of items
    total_items = data.groupby('ItemID')

    # Splitting the dataset
    train, test = train_test_split(data, test_size=0.2, random_state=22)

    # Training set
    train_clicks = len(train)
    train_sessions = len(train.groupby('SessionID'))
    train_items = len(train.groupby('ItemID'))

    properties['clicks_tr'] = train_clicks
    properties['sessions_tr'] = train_sessions
    properties['items_tr'] = train_items
    properties['average_session_length_tr'] = train_clicks / train_sessions
    properties['average_freq_item_tr'] = train_clicks / train_items

    # Testing set
    test_clicks = len(test)
    test_sessions = len(test.groupby('SessionID'))
    test_items = len(test.groupby('ItemID'))
    properties['clicks_ts'] = test_clicks
    properties['sessions_ts'] = test_sessions
    properties['items_ts'] = test_items
    properties['average_session_length_ts'] = test_clicks / test_sessions
    properties['average_freq_item_ts'] = test_clicks / test_items

    # Create new dataset with the extracted features
    features = [[train_items, train_sessions, train_clicks / train_sessions, train_clicks / train_items,
                test_clicks / test_sessions, test_clicks / test_items]]

    features_df = pd.DataFrame(features, columns=['n_items', 'n_sessionsTr', 'avgLen_sessionsTr', 'avgFreq_itemsTr',
                                                  'avgLen_sessionsTe', 'avgFreq_itemsTe'])
    model = predict(features_df)
    properties['model'] = model

    return properties
