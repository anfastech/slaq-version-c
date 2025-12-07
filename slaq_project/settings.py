# slaq_project/settings.py
import os
from pathlib import Path
from datetime import timedelta

from environ import Env
import dj_database_url

env = Env()
BASE_DIR = Path(__file__).resolve().parent.parent
Env.read_env(BASE_DIR / '.env')

ENVIRONMENT = env('ENVIRONMENT', default='production')

# Debug mode
# In production, DEBUG must be False for security. Use ENVIRONMENT to control this.
if ENVIRONMENT == 'development':
    DEBUG = True
else:
    DEBUG = False

ALLOWED_HOSTS = ['*', 'localhost', '127.0.0.1']

SECRET_KEY = env('DJANGO_SECRET_KEY')
ENCRYPT_KEY = env('DJANGO_ENCRYPT_KEY')

SUPABASE_URL = env.str('SUPABASE_URL')
SUPABASE_ANON_KEY = env.str('SUPABASE_ANON_KEY')
SUPABASE_SERVICE_ROLE_KEY = env.str('SUPABASE_SERVICE_ROLE_KEY')
SUPABASE_BUCKET_NAME = env.str('SUPABASE_BUCKET_NAME')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',

    # WhiteNoise app (disables Django's own static file handling in runserver)
    'whitenoise.runserver_nostatic',
    'django.contrib.staticfiles',

    # Third party
    'django_celery_results',

    # Local apps
    'core.apps.CoreConfig',
    'diagnosis.apps.DiagnosisConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',

    # WhiteNoise middleware must be directly after SecurityMiddleware
    'whitenoise.middleware.WhiteNoiseMiddleware',

    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'slaq_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'slaq_project.wsgi.application'

# Database development
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('DB_NAME'),
        'USER': env('DB_USER'),
        'PASSWORD': env('DB_USER_PASSWORD'),
        'HOST': env('DB_HOST'),
        'PORT': env('DB_PORT'),
    }
}

POSTGRES_LOCALLY = False
if ENVIRONMENT == 'production' or POSTGRES_LOCALLY:
    DATABASES['default'] = dj_database_url.parse(env('DATABASE_URL'))

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

# Use WhiteNoise for efficient static file serving in production
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files (User uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Supabase Storage Configuration
if ENVIRONMENT == 'production':
    DEFAULT_FILE_STORAGE = 'core.supabase_storage.SupabaseStorage'

# Default primary key field type

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Authentication URLs
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/'

# File Upload Settings (MVP: max 10MB)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_AUDIO_FORMATS = ['.wav', '.mp3', '.webm', '.ogg']

# Celery Configuration
CELERY_BROKER_URL = env('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = 'django-db'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# AI Model Configuration
AI_MODELS_DIR = BASE_DIR / 'ml_models'
WAV2VEC2_BASE_MODEL = "facebook/wav2vec2-base-960h"

# Audio Processing Settings
AUDIO_SAMPLE_RATE = 16000

# Stutter Detection Thresholds
STUTTER_THRESHOLDS = {
    'prolongation_duration': 0.4,  # seconds
    'mild_mismatch': 10,  # percentage
    'moderate_mismatch': 25,
    'severe_mismatch': 50,
}

# AI Engine API Configuration
# Base URL for the external stutter detection API (HuggingFace Space)
# The /analyze endpoint will be appended automatically in detect_stuttering.py
STUTTER_API_URL = env('STUTTER_API_URL', default='https://anfastech-slaq-version-c-ai-enginee.hf.space')
# Timeout for API requests in seconds (default: 5 minutes for long audio files)
STUTTER_API_TIMEOUT = env.int('STUTTER_API_TIMEOUT', default=300)
# Default language for analysis if not specified
DEFAULT_LANGUAGE = env('DEFAULT_LANGUAGE', default='hindi')
# Maximum retries for API requests
STUTTER_API_MAX_RETRIES = env.int('STUTTER_API_MAX_RETRIES', default=3)
# Retry delay in seconds
STUTTER_API_RETRY_DELAY = env.int('STUTTER_API_RETRY_DELAY', default=5)


ACCOUNT_USERNAME_BLACKLIST = ['admin', 'administrator', 'root', 'superuser', 'staff', 'user', 'test', 'username', 'theboss']