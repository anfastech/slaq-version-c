# diagnosis/urls.py
from django.urls import path
from . import views

app_name = 'diagnosis'

urlpatterns = [
    # Recording
    path('record/', views.record_audio, name='record'),
    path('upload/', views.upload_recording, name='upload'),
    
    # Recordings List
    path('recordings/', views.recordings_list, name='recordings_list'),
    path('recordings/<int:recording_id>/', views.recording_detail, name='recording_detail'),
    path('recordings/<int:recording_id>/delete/', views.delete_recording, name='delete_recording'),
    
    # Analysis
    path('analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),
    
    # API Endpoints (for AJAX)
    path('api/status/<int:recording_id>/', views.check_status, name='check_status'),
]
