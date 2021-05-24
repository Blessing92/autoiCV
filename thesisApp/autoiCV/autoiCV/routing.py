from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from autoapp import consumer
from django.urls import path, re_path


websocket_urlPattern=[
    path('ws/polData/', consumer.DashConsumer.as_asgi()),
    path('ws/some_url/', consumer.WSConsumer.as_asgi()),
    re_path(r'ws/chat/(?P<room_name>\w+)/$', consumer.ChatRoomConsumer.as_asgi()),

]

application=ProtocolTypeRouter({
    'websocket':AuthMiddlewareStack(URLRouter(websocket_urlPattern))
})