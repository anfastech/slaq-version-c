# 🚀 SLAQ AI Speech Diagnosis — Full System Setup Guide

A clean, simplified, and production-ready README for running the full system: Redis, Celery, Django, and AI model downloads.
Whole development using production level resources. 
---

# ⚡ 0. Initial Setup

## 📦 Install Python Environment

```sh
python -m venv venv
venv\Scripts\activate
pip install celery dotenv  django  django_celery_results psycopg2-binary librosa fastdtw redis
pip install --upgrade pip
```

## 📥 Install Required Libraries

Before downloading models:

```sh
pip install transformers torch torchaudio
```

---

# ⚡ 1. Download AI Models (One-time Only)

This step downloads ~3GB of audio AI models.

**Run this command in Terminal 1:**

```sh
python download_model.py
```

Models are stored in:

```
C:\Users\<YOUR_USERNAME>\.cache\huggingface\hub
```

---

# ⚡ 2. Start System Services

# 2.1 🚚 Redis Server

Redis is required for Celery background task processing.

## ✔ Ensure Redis is Installed

Install using the official MSI installer (recommended for Windows):
👉 [https://github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases)
Download:

```
Redis-x64-3.2.100.msi
```

During installation:

* ✔ Add Redis to PATH
* ✔ Install as a Windows Service

## ✔ Start Redis

```sh
redis-server
```

Or start Windows service:

```sh
net start Redis
```

---

# 2.2 🔁 Celery Worker

Celery executes the AI analysis jobs (audio processing, ML pipeline).

## ✔ Start Celery (Terminal 2)

### 1. Activate Virtual Environment

```sh
venv\Scripts\activate
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run Celery

```sh
pip install eventlet
celery -A slaq_project worker --pool=eventlet -l info
```

---

# ⚡ 3. Start Django Server

The main backend web server.

## ✔ Steps (Terminal 3)

### 1. Activate Environment

```sh
venv\Scripts\activate
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run Django

```sh
python manage.py runserver
```

Your app is now available at:

```
http://127.0.0.1:8000/
```

---

# 📝 Notes

* Always keep **Redis running** before starting Celery or Django.
* Celery and Django must run in **separate terminals**.
* If models aren’t found, delete HuggingFace cache and re-run the download script.
* Keep `requirements.txt` updated.

---

# 🎉 System Overview

| Component              | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| **Redis**              | Message broker for Celery                |
| **Celery Worker**      | Runs background AI analysis              |
| **Django Server**      | Backend API and business logic           |
| **HuggingFace Models** | Speech processing + stuttering detection |

---

# ❤️ Author

Clean & production-ready setup guide generated with the help of AI.

Ready to deploy. 🚀
