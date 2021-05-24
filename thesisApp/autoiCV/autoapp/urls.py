from django.conf.urls import url
from django.urls import path
from autoapp import views

# TEMPLATE TAGGING
app_name = 'autoapp'

urlpatterns = [
    path('other/', views.other, name='other'),
    path('base/', views.base, name='base'),
    path('training/', views.selected_model, name='training'),
    path('user_login/', views.user_login, name='user_login'),
    path('contact_upload/', views.contact_upload, name='contact_upload'),
    path('upload_csv/', views.upload_csv, name='upload_csv'),
    path('<str:room_name>/', views.room, name='room')

]