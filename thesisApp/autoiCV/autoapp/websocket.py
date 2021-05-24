import json
import requests
import redis
import websocket
import random, time

ws = websocket.WebSocket()
ws.connect('ws://localhost/8000/ws/polData/')