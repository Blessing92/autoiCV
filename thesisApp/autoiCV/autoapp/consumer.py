import os
import time
import json
import argparse
import datetime
import pandas as pd
from random import randint
from time import sleep
from autoapp.utils import spop_parse_args
from autoapp.spop import SessionPop
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.generic.websocket import WebsocketConsumer





class DashConsumer(AsyncWebsocketConsumer):
    
    async def connect(self):
        self.groupname='dashboard'
        await self.channel_layer.group_add(
            self.groupname,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):

        #await self.disconnect()
        pass

    async def receive(self, text_data):
        print('>>>', text_data)
        pass


class WSConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

        self.send(json.dumps({'message': "Done initializing hyperparameters..."}))

        args = spop_parse_args()

        train_data = os.path.join(args.data_folder, args.train_data)
        x_train = pd.read_csv(train_data)
        valid_data = os.path.join(args.data_folder, args.valid_data)
        x_valid = pd.read_csv(valid_data)
        x_valid.sort_values(args.sessionid, inplace=True)


        self.send(json.dumps({'message': "Finished Reading Data \nStart Model Fitting..."}))
        # Fitting AR Model
        t1 = time.time()
        model = SessionPop(top_n = args.topn, session_key = args.sessionid, item_key = args.itemid)
        model.fit(x_train)
        t2 = time.time()
        message = 'End Model Fitting with total time = '+ str(t2 - t1) + '\n Start Predictions...'
        self.send(json.dumps({'message': message}))

        # Test Set Evaluation
        test_size = 0.0
        hit = 0.0
        MRR = 0.0
        cur_length = 0
        cur_session = -1
        last_items = []
        t1 = time.time()
        index_item = x_valid.columns.get_loc(args.itemid)
        index_session = x_valid.columns.get_loc(args.sessionid)
        train_items = model.items
        counter = 0
        for row in x_valid.itertuples( index=False ):
            counter += 1
            if counter % 5000 == 0:
                message = str(datetime.datetime.now()) + '--> Finished Prediction for '+ str(counter) + ' items.'
                self.send(json.dumps({'message': message}))

            session_id, item_id = row[index_session], row[index_item]
            if session_id != cur_session:
                cur_session = session_id
                last_items = []
                cur_length = 0
            
            if item_id in train_items:
                if len(last_items) > cur_length: #make prediction
                    cur_length += 1
                    test_size += 1
                    # Predict the most similar items to items
                    predictions = model.predict_next(last_items, k = args.K)
                    # Evaluation
                    rank = 0
                    for predicted_item in predictions:
                        rank += 1
                        if predicted_item == item_id:
                            hit += 1.0
                            MRR += 1/rank
                            break
                
                last_items.append(item_id)
        t2 = time.time()
        # print('Recall: {}'.format(hit / test_size))
        # print ('\nMRR: {}'.format(MRR / test_size))
        
        message = 'End Model Predictions with total time = ' + str(t2 - t1)
        self.send(json.dumps({'message': message}))

        for i in range(len(last_items)):
            message = 'These are the predicted items: ' + str(last_items[i])
            self.send(json.dumps({'message': message}))
    
    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        self.send(text_data=json.dumps({
            'message': message
        }))
    


class ChatRoomConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.room_name

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        username = text_data_json['username']

        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chatroom_message',
                'message': message,
                'username': username,
            }
        )

    async def chatroom_message(self, event):
        message = event['message']
        username = event['username']

        await self.send(text_data=json.dumps({
            'message': message,
            'username': username,
        }))

    pass


