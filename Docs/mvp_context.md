┌─────────────────────────────────────────────────────────────┐
│          SLAQ MVP - Django Monolithic Application            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Presentation Layer (Templates + Static Files)       │   │
│  │  - HTML Templates (Django Template Language)         │   │
│  │  - Tailwind CSS (Responsive Design)                  │   │
│  │  - Vanilla JavaScript (Audio Recording, AJAX)        │   │
│  │  - Chart.js (Data Visualization)                     │   │
│  └────────────────────┬────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │  Views Layer (Django Views)                          │  │
│  │  - Session Authentication                             │  │
│  │  - Audio Upload Handling                             │  │
│  │  - Analysis Results Display                          │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │  Business Logic                                       │  │
│  │  - Audio Validation                                   │  │
│  │  - Celery Task Queue                                 │  │
│  └────────────────────┬─────────────────────────────────┘  │
└────────────────────────┼────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│  PostgreSQL DB   │            │  Celery Worker   │
│  - Users         │            │  ┌────────────┐  │
│  - Patients      │            │  │ AI Engine  │  │
│  - Recordings    │            │  │ Wav2Vec2   │  │
│  - Analyses      │            │  └────────────┘  │
└──────────────────┘            └──────────────────┘
                                        │
                                        ▼
                                ┌──────────────────┐
                                │  Redis Queue     │
                                └──────────────────┘

═══════════════════════════════════════════════════════════════
                    MVP SCOPE & REQUIREMENTS
═══════════════════════════════════════════════════════════════

## 🎯 MVP Goal
Build a functional stuttering analysis platform where patients can record audio, receive AI-powered stutter detection analysis, and view detailed results with visualizations.

## ✅ MVP Features (Phase 1)

### 1. User Authentication
- ✅ Patient registration with email/password
- ✅ Login/logout functionality
- ✅ Session-based authentication
- ✅ Basic profile page (view only)
- ❌ Password reset (future)
- ❌ Email verification (future)
- ❌ Therapist accounts (future)

### 2. Audio Recording & Upload
- ✅ Browser-based audio recording (Web Audio API)
- ✅ Visual feedback during recording (waveform)
- ✅ Record/Stop/Play preview controls
- ✅ File upload from device (alternative to recording)
- ✅ Audio format validation (WAV, MP3, WebM)
- ✅ File size limit (max 10MB)
- ❌ Real-time recording quality feedback (future)
- ❌ Noise cancellation (future)

### 3. AI Analysis
- ✅ Wav2Vec2 speech-to-text transcription
- ✅ Character mismatch detection
- ✅ CTC loss score calculation
- ✅ Severity classification (None/Mild/Moderate/Severe)
- ✅ Confidence score
- ✅ Background processing via Celery
- ✅ Status tracking (pending → processing → completed)
- ❌ Stutter event timestamps (future)
- ❌ Event type classification (future)
- ❌ Advanced ML models (future)

### 4. Results Display
- ✅ Analysis results page with key metrics
- ✅ Severity badge (color-coded)
- ✅ Actual vs Target transcript comparison
- ✅ Mismatch percentage display
- ✅ Basic visualization (Chart.js bar/line charts)
- ✅ List of mismatched character sequences
- ❌ Interactive audio playback with timestamp markers (future)
- ❌ Detailed stutter event timeline (future)
- ❌ PDF report download (future)

### 5. Recording History
- ✅ List all patient recordings with status
- ✅ Filter by status (all/completed/pending)
- ✅ View individual recording details
- ✅ Delete recordings
- ❌ Advanced search/filtering (future)
- ❌ Bulk operations (future)

### 6. Basic Dashboard
- ✅ Overview with recent recordings count
- ✅ Latest analysis summary
- ✅ Quick access to record new audio
- ❌ Progress charts over time (future)
- ❌ Improvement metrics (future)
- ❌ Therapy recommendations (future)

## 🏗️ MVP Technology Stack

### Backend (Core MVP)
- **Framework**: Django 4.2
- **Database**: PostgreSQL 13+ (can use SQLite for local dev)
- **Task Queue**: Celery 5.x + Redis
- **Authentication**: Django Session Auth
- **AI Model**: wav2vec2-base-960h (Hugging Face)

### Frontend (Simplified)
- **Templates**: Django Templates (DTL)
- **CSS**: Tailwind CSS via CDN (no build process)
- **JavaScript**: Vanilla JS (ES6+)
- **Audio**: Web Audio API + MediaRecorder
- **Charts**: Chart.js 4.x via CDN

### Infrastructure (Minimal)
- **Web Server**: Django development server (MVP), Gunicorn (production)
- **Cache/Queue**: Redis 6+
- **File Storage**: Local filesystem (media directory)

## 📊 MVP Database Models

### Core Models (Simplified)
```python
# core/models.py
- User (Django built-in)
- Patient
  - user (OneToOne)
  - date_of_birth
  - phone_number (optional)
  - created_at
```

### Diagnosis Models (MVP Scope)
```python
# diagnosis/models.py
- AudioRecording
  - patient (FK)
  - audio_file
  - status (pending/processing/completed/failed)
  - duration_seconds
  - recorded_at
  - processed_at

- AnalysisResult (Simplified)
  - recording (OneToOne)
  - actual_transcript
  - target_transcript
  - mismatched_chars (JSON list)
  - mismatch_percentage
  - ctc_loss_score
  - severity (none/mild/moderate/severe)
  - confidence_score
  - created_at
```

**Excluded for MVP:**
- StutterEvent model (detailed events)
- Report model
- TherapyRecommendation model
- ProgressTracking model

## 🔄 MVP User Flow

### 1. Registration & Login
```
User visits landing page
→ Click "Register"
→ Fill registration form (email, password, name, DOB)
→ System creates User + Patient profile
→ Auto-login after registration
→ Redirect to dashboard
```

### 2. Record Audio
```
Patient clicks "Record New Audio"
→ Browser requests microphone permission
→ Patient speaks into microphone
→ Visual waveform shows audio input
→ Click "Stop" when finished
→ Preview recording (play button)
→ Click "Upload for Analysis"
→ File uploaded to server
→ Recording status: "Pending"
→ Redirect to recordings list
```

### 3. AI Analysis (Background)
```
Celery worker picks up task
→ Load Wav2Vec2 model
→ Transcribe audio
→ Compare with target transcript
→ Calculate metrics (mismatch %, CTC loss)
→ Classify severity
→ Save AnalysisResult
→ Update recording status: "Completed"
```

### 4. View Results
```
Patient sees "Completed" badge on recording
→ Click to view analysis
→ See results page:
  - Severity badge (color-coded)
  - Mismatch percentage
  - Actual vs Target transcripts
  - Basic metrics chart
  - List of mismatched sequences
```

## 🎨 MVP Pages & Templates

### Authentication Pages
1. **Landing Page** (`home.html`)
   - Hero section with app description
   - "Get Started" and "Login" buttons
   - Simple, clean design

2. **Register Page** (`register.html`)
   - Form: Email, Password, Full Name, Date of Birth
   - Validation messages
   - Link to login page

3. **Login Page** (`login.html`)
   - Email/password form
   - "Remember me" checkbox
   - Link to register page

### Main Application Pages
4. **Dashboard** (`dashboard.html`)
   - Welcome message with patient name
   - Statistics cards:
     - Total recordings
     - Completed analyses
     - Latest severity
   - "Record New Audio" CTA button
   - Recent recordings list (last 5)

5. **Record Audio Page** (`record.html`)
   - Microphone permission request
   - Record/Stop/Play controls
   - Waveform visualization (canvas)
   - File upload alternative
   - Upload button

6. **Recordings List** (`recordings_list.html`)
   - Table/card view of all recordings
   - Columns: Date, Duration, Status, Actions
   - Status badges (pending/processing/completed/failed)
   - Click to view analysis (if completed)
   - Delete button

7. **Analysis Results** (`analysis_detail.html`)
   - Recording info (date, duration)
   - Severity badge (large, prominent)
   - Key metrics:
     - Mismatch percentage (with chart)
     - CTC loss score
     - Confidence score
   - Transcript comparison:
     - Target transcript
     - Actual transcript (highlighted differences)
   - Mismatched sequences list
   - Basic bar chart showing metrics

8. **Profile Page** (`profile.html`)
   - Display patient info
   - Email, name, DOB, phone
   - Join date
   - ❌ Edit functionality (future)

## 🔌 MVP URL Structure

```python
# Authentication
GET  /                          # Landing page
GET  /register/                 # Registration form
POST /register/                 # Process registration
GET  /login/                    # Login form
POST /login/                    # Process login
GET  /logout/                   # Logout

# Main App
GET  /dashboard/                # Patient dashboard
GET  /profile/                  # View profile

# Audio Recording
GET  /record/                   # Recording interface
POST /recordings/upload/        # Upload audio file
GET  /recordings/               # List all recordings
GET  /recordings/<id>/          # View recording details
POST /recordings/<id>/delete/   # Delete recording

# Analysis
GET  /analysis/<id>/            # View analysis results
GET  /api/analysis/<id>/status/ # Check status (AJAX)
```

## 🎨 MVP UI/UX Design

### Color Theme 
- White && Black 
- Dark Green (#009050) 

### Color Scheme (Severity)
- **None**: Green (#10b981) - No stuttering detected
- **Mild**: Yellow (#fbbf24) - Minor stuttering
- **Moderate**: Orange (#f97316) - Noticeable stuttering
- **Severe**: Red (#ef4444) - Significant stuttering

### Layout
- **Mobile First**: Responsive design starting at 320px
- **Clean & Simple**: Minimal distractions
- **Clear CTAs**: Prominent action buttons
- **Status Feedback**: Clear loading states and success/error messages

### Tailwind Components
- **Buttons**: `btn-primary`, `btn-secondary`, `btn-danger`
- **Cards**: White background, shadow, rounded corners
- **Forms**: Full-width inputs, clear labels, validation styling
- **Badges**: Pill-shaped status indicators
- **Tables**: Striped rows, hover effects

## 💾 MVP Static Files Structure

```
static/
├── css/
│   ├── tailwind.min.css         # Tailwind via CDN (MVP)
│   └── custom.css               # Custom overrides (minimal)
├── js/
│   ├── audio-recorder.js        # Web Audio API logic
│   ├── charts.js                # Chart.js initialization
│   └── utils.js                 # Helper functions
└── images/
    └── logo.png                 # App logo
```

## 🚀 MVP Development Timeline

### Week 1: Setup & Authentication
- [ ] Django project setup
- [ ] PostgreSQL database configuration
- [ ] User registration/login
- [ ] Base templates (base.html, navbar, footer)
- [ ] Landing page
- [ ] Basic styling with Tailwind

### Week 2: Audio Recording
- [ ] Audio recording page UI
- [ ] Web Audio API integration
- [ ] MediaRecorder implementation
- [ ] File upload endpoint
- [ ] Audio validation
- [ ] Recordings list page

### Week 3: AI Integration
- [ ] Celery + Redis setup
- [ ] Wav2Vec2 model integration
- [ ] Analysis task implementation
- [ ] AnalysisResult model
- [ ] Status tracking
- [ ] Error handling

### Week 4: Results Display
- [ ] Analysis detail page
- [ ] Chart.js integration
- [ ] Transcript comparison UI
- [ ] Dashboard page
- [ ] Profile page
- [ ] Testing & bug fixes

### Week 5: Polish & Deployment
- [ ] Responsive design testing
- [ ] Loading states & feedback
- [ ] Error messages
- [ ] Performance optimization
- [ ] Documentation
- [ ] MVP deployment

## 🧪 MVP Testing Checklist

### Functional Testing
- [ ] User can register and login
- [ ] User can record audio via microphone
- [ ] User can upload audio file
- [ ] Audio is validated (format, size)
- [ ] Celery task processes audio
- [ ] Analysis results are displayed correctly
- [ ] Severity is classified accurately
- [ ] User can view recording history
- [ ] User can delete recordings
- [ ] Status updates work correctly

### UI/UX Testing
- [ ] Responsive on mobile (320px+)
- [ ] Responsive on tablet (768px+)
- [ ] Responsive on desktop (1024px+)
- [ ] Loading indicators show during processing
- [ ] Success/error messages display
- [ ] Forms validate input
- [ ] Navigation works correctly

### Browser Testing
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest - Mac/iOS)
- [ ] Edge (latest)

## 📦 MVP Dependencies (requirements.txt)

```txt
# Core Django
Django==4.2.7
psycopg2-binary==2.9.9
python-decouple==3.8

# Task Queue
celery==5.3.4
redis==5.0.1

# AI/ML
torch==2.1.0
transformers==4.35.0
librosa==0.10.1
soundfile==0.12.1

# Utilities
Pillow==10.1.0
python-dateutil==2.8.2

# Production (optional for MVP)
gunicorn==21.2.0
whitenoise==6.6.0
```

## ⚙️ MVP Configuration

### settings.py (Essential Settings)
```python
# Security (Development)
DEBUG = True
SECRET_KEY = 'your-secret-key-here'
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'slaq_mvp',
        'USER': 'postgres',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Static & Media
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Celery
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# File Upload
MAX_UPLOAD_SIZE = 10485760  # 10MB
ALLOWED_AUDIO_FORMATS = ['.wav', '.mp3', '.webm', '.ogg']
```

## 🚀 MVP Deployment (Simple)

### Local Development
```bash
# Terminal 1: Django
python manage.py runserver

# Terminal 2: Celery Worker
celery -A slaq_project worker --pool=solo -l info

# Terminal 3: Redis
redis-server
```

### Production (Basic)
```bash
# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Run migrations
python manage.py migrate

# Start Gunicorn
gunicorn slaq_project.wsgi:application --bind 0.0.0.0:8000

# Start Celery (separate process)
celery -A slaq_project worker -l info

# Use Nginx as reverse proxy (recommended)
```

## 🎯 MVP Success Criteria

### Functional
- ✅ Users can register and login
- ✅ Users can record and upload audio
- ✅ AI analysis completes within 2 minutes
- ✅ Results display with 95%+ accuracy
- ✅ No critical bugs or crashes

### Performance
- ✅ Page load < 3 seconds
- ✅ Audio upload < 10 seconds
- ✅ Responsive on all devices
- ✅ Supports 10+ concurrent users

### User Experience
- ✅ Intuitive navigation
- ✅ Clear feedback and status updates
- ✅ Accessible on mobile devices
- ✅ Professional, clean design

## 🚫 MVP Exclusions (Future Versions)

### Not in MVP
- ❌ Therapist accounts and management
- ❌ Email notifications
- ❌ Password reset functionality
- ❌ Profile editing
- ❌ Advanced search and filters
- ❌ PDF report generation
- ❌ Progress tracking over time
- ❌ Therapy recommendations
- ❌ Social features
- ❌ Mobile app
- ❌ Real-time collaboration
- ❌ Payment/billing
- ❌ Multi-language support
- ❌ Detailed stutter event timeline
- ❌ Advanced ML models
- ❌ Voice biometrics

These features are planned for future phases after MVP validation.

---

## 📞 MVP Support

### Development Team
- **Role**: Full-Stack Developer
- **Focus**: Build core features, integrate AI model, design UI

### Key Contacts
- **Technical Questions**: Check Django/Celery documentation
- **AI Model**: Hugging Face Wav2Vec2 docs
- **Deployment**: Django deployment guide

---

**MVP Version**: 1.0.0  
**Target Launch**: [Your Date]  
**Status**: In Development  
**Last Updated**: November 9, 2025
