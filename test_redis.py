# test_redis.py
import os
import django
from django.conf import settings

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'slaq_project.settings')
django.setup()

from celery import Celery

# Get Redis URL from Django settings
redis_url = settings.CELERY_BROKER_URL

# Test connection
app = Celery('test')
app.conf.broker_url = redis_url
app.conf.result_backend = redis_url

try:
    connection = app.connection()
    connection.ensure_connection(max_retries=3)
    print("✅ Redis connection successful!")
    print(f"Connected to: {redis_url}")
    
    # Optional: Test if we can send a message
    with connection.channel() as channel:
        print("✅ Channel creation successful!")
        
except Exception as e:
    print(f"❌ Redis connection failed: {e}")