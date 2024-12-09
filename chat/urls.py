from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('chat/', views.chat_page, name='chat'),
    path('process_message/', views.process_message, name='process_message'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
]