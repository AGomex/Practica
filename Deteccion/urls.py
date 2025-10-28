from django.contrib import admin
from django.urls import path
from .views import video_feed, home

urlpatterns = [
    path("", home, name="home"),
    path("video", video_feed, name="video_feed"),
]