# SLAQ MVP - Quick Start Guide 🚀

## Prerequisites
- Python 3.10+
- PostgreSQL 13+ (or use SQLite for testing)
- Redis Server
- pip (Python package manager)

## Installation & Setup

### 1. Install Dependencies
```bash
cd "C:\Users\Faheem\Desktop\AI\Slaq\slaq_project"
pip install -r requirements.txt
```

### 2. Database Setup

#### Option A: PostgreSQL (Recommended for Production)
```bash
# Make sure PostgreSQL is running
# Create database (if not exists)
createdb slaq_db

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

#### Option B: SQLite (Quick Testing)
Edit `slaq_project/settings.py` and temporarily replace PostgreSQL config:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```
Then run:
```bash
python manage.py migrate
python manage.py createsuperuser
```

### 3. Start Services

#### Terminal 1: Redis Server
```bash
redis-server
```

#### Terminal 2: Celery Worker
```bash
cd "C:\Users\Faheem\Desktop\AI\Slaq\slaq_project"
celery -A slaq_project worker -l info
```

#### Terminal 3: Django Development Server
```bash
cd "C:\Users\Faheem\Desktop\AI\Slaq\slaq_project"
python manage.py runserver
```

### 4. Access the Application
Open your browser and navigate to:
- **Homepage**: http://localhost:8000/
- **Register**: http://localhost:8000/register/
- **Login**: http://localhost:8000/login/
- **Admin**: http://localhost:8000/admin/

## Testing the Application

### 1. Register a Patient Account
1. Go to http://localhost:8000/
2. Click "Get Started" or "Register"
3. Fill in the registration form:
   - Username
   - Email
   - First Name, Last Name
   - Date of Birth
   - Phone Number (optional)
   - Password + Confirmation
4. Click "Create Account"
5. You'll be auto-logged in and redirected to dashboard

### 2. Record Audio
1. From dashboard, click "Record New Audio"
2. **Option A - Microphone Recording**:
   - Click "Start Recording"
   - Allow microphone access
   - Speak for 10-30 seconds
   - Click "Stop Recording"
   - Click "Play" to review (optional)
   - Click "Upload & Analyze"
3. **Option B - File Upload**:
   - Scroll to "Upload an Audio File"
   - Choose a WAV/MP3/M4A file
   - Click "Upload File"

### 3. View Analysis Results
- Wait for processing (auto-refresh every 5 seconds)
- View recording details
- Click "View Analysis" when complete
- Explore:
  - Metrics dashboard
  - Transcript comparison
  - Stutter events timeline
  - Charts and visualizations

### 4. Explore Features
- **Dashboard**: View stats and recent recordings
- **Profile**: View personal information
- **Recordings List**: Filter by status, download, delete
- **Recording Detail**: Play audio, view analysis
- **Analysis Detail**: Full metrics and charts

## Troubleshooting

### Migrations Not Running
```bash
# Check for errors
python manage.py check

# Show migration status
python manage.py showmigrations

# Make migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate --run-syncdb
```

### Celery Not Working
```bash
# Check Redis is running
redis-cli ping  # Should return "PONG"

# Start Celery with verbose logging
celery -A slaq_project worker -l debug

# On Windows, you might need:
celery -A slaq_project worker -l info --pool=solo
```

### Static Files Not Loading
```bash
# Collect static files
python manage.py collectstatic --no-input

# Check STATIC_URL in settings
# Make sure DEBUG=True for development
```

### Audio Recording Not Working
- Check browser permissions (allow microphone access)
- Use HTTPS or localhost (HTTP won't work for microphone)
- Try different browsers (Chrome/Firefox recommended)
- Check console for JavaScript errors

### Database Connection Error
If using PostgreSQL:
```bash
# Check PostgreSQL is running
pg_isready

# Check connection settings in settings.py
# Verify DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

# Or switch to SQLite for testing (see Option B above)
```

## Project Structure

```
slaq_project/
├── manage.py
├── requirements.txt
├── slaq_project/         # Main project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── core/                 # Authentication & dashboard
│   ├── models.py         # Patient model
│   ├── views.py          # Auth views
│   ├── forms.py          # Registration form
│   └── urls.py
├── diagnosis/            # Recording & analysis
│   ├── models.py         # AudioRecording, AnalysisResult
│   ├── views.py          # Recording views
│   ├── forms.py          # Upload form
│   ├── urls.py
│   ├── tasks.py          # Celery tasks
│   └── ai_engine/        # AI model code
├── templates/            # HTML templates
│   ├── base.html
│   ├── components/
│   ├── core/
│   └── diagnosis/
└── static/              # CSS & JavaScript
    ├── css/
    └── js/
```

## Environment Variables (Optional)

Create a `.env` file in project root:
```env
DEBUG=True
SECRET_KEY=your-secret-key-here
DB_NAME=slaq_db
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432
CELERY_BROKER_URL=redis://localhost:6379/0
```

Then install python-decouple:
```bash
pip install python-decouple
```

## Default Admin Access
After creating superuser:
- URL: http://localhost:8000/admin/
- Username: (your superuser username)
- Password: (your superuser password)

## Common Commands

```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver

# Run Celery worker
celery -A slaq_project worker -l info

# Collect static files
python manage.py collectstatic

# Run Django shell
python manage.py shell

# Check for issues
python manage.py check

# Show migrations
python manage.py showmigrations
```

## Testing Checklist

- [ ] Can access homepage (/)
- [ ] Can register new account
- [ ] Can login with credentials
- [ ] Can view dashboard with stats
- [ ] Can view profile
- [ ] Can record audio from microphone
- [ ] Can upload audio file
- [ ] Can see recording in list
- [ ] Can filter recordings by status
- [ ] Can view recording details
- [ ] Can play audio in browser
- [ ] Analysis completes (Celery working)
- [ ] Can view analysis results
- [ ] Charts render correctly
- [ ] Can delete recording
- [ ] Can logout
- [ ] Mobile navigation works
- [ ] Forms show validation errors

## Support & Documentation

- **Full Documentation**: `Docs/IMPLEMENTATION_SUMMARY.md`
- **Architecture**: `Docs/context.md`
- **MVP Scope**: `Docs/mvp_context.md`
- **Development Plan**: `Docs/MVP_DEVELOPMENT_PLAN.md`

## Next Steps

1. ✅ Complete database setup
2. ✅ Test basic user flow
3. ⏳ Configure AI model (Wav2Vec2)
4. ⏳ Test Celery task processing
5. ⏳ Deploy to production server
6. ⏳ Add SSL certificate
7. ⏳ Configure domain name

---

**Status**: 🟢 Ready for Testing!

The SLAQ MVP is complete and ready to run. Follow the steps above to set up your development environment and start testing the application.

For questions or issues, refer to the troubleshooting section or check the detailed documentation in the `Docs/` folder.
