┌─────────────────────────────────────────────────────────────┐
│          Django Monolithic Application (Port 8000)          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Presentation Layer (Templates + Static Files)      │    │
│  │  - HTML Templates (Django Template Language)        │    │
│  │  - Tailwind CSS (Responsive Design)                 │    │
│  │  - Vanilla JavaScript (Audio Recording, AJAX)       │    │
│  │  - Chart.js (Data Visualization)                    │    │
│  └────────────────────┬────────────────────────────────┘    │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  Views Layer (Django Views)                          │   │
│  │  - Session Authentication  (With Username)           │   │
│  │  - Audio Upload Handling                             │   │
│  │  - Report Generation Views                           │   │
│  │  - AJAX API Endpoints (JSON responses)               │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  Business Logic Layer                                │   │
│  │  - Audio Validation                                  │   │
│  │  - Celery Task Queue Management                      │   │
│  │  - Report Generation Logic                           │   │
│  └────────────────────┬─────────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐            ┌──────────────────┐
│  PostgreSQL DB   │            │  Celery Worker   │
│  - User Data     │            │  ┌────────────┐  │
│  - Recordings    │            │  │ AI Engine  │  │
│  - Analyses      │            │  │ Wav2Vec2   │  │
│  - Reports       │            │  │ Models     │  │
└──────────────────┘            │  └────────────┘  │
                                │  - Async Tasks   │
                                │  - Model Loading │
                                └──────────────────┘
                                        │
                                        ▼
                                ┌──────────────────┐
                                │  Redis Queue     │
                                │  - Task Queue    │
                                │  - Results Cache │
                                └──────────────────┘

═══════════════════════════════════════════════════════════════
                    PROJECT CONTEXT & OVERVIEW
═══════════════════════════════════════════════════════════════

## 🎯 Project Purpose
SLAQ is an AI-powered stuttering diagnosis and therapy tracking system that helps speech therapists monitor patient progress and provides personalized therapy recommendations using advanced machine learning models.

## 🏗️ Technology Stack

### Backend
- **Framework**: Django 4.x (Monolithic Architecture)
- **Database**: PostgreSQL (production & local development) 
- **Task Queue**: Celery + Redis for asynchronous AI processing
- **Authentication**: Django Session-based Authentication with CSRF protection
- **AI Model**: Wav2Vec2 (Facebook's speech recognition model)
- **File Storage**: Django FileField with organized media structure
- **Template Engine**: Django Template Language (DTL)

### Frontend (Integrated with Django)
- **Templates**: Django Templates (server-side rendering)
- **Styling**: Tailwind CSS (Responsive - Mobile, Tablet, Desktop)
- **JavaScript**: Vanilla JavaScript for interactivity
- **Audio Recording**: Web Audio API and MediaRecorder API
- **Charts/Visualization**: Chart.js for progress charts and analytics
- **AJAX**: Fetch API for asynchronous data loading without page refresh
- **Static Files**: Organized CSS/JS in `/static/` directory

### AI/ML Layer
- **Model**: wav2vec2-base-960h (Hugging Face Transformers)
- **Processing**: 
  - Speech-to-text transcription
  - CTC (Connectionist Temporal Classification) loss calculation
  - Character mismatch detection
  - Stutter event classification
  - Timestamp extraction

## 📋 Key Features

### 1. Audio Recording & Upload
- Patient records speech samples through web interface using browser's microphone
- Real-time audio visualization during recording
- Validates audio format (WAV, MP3, WebM)
- Client-side validation before upload
- Stores files in organized directory structure: `recordings/{patient_id}/{year}/{month}/`

### 2. AI-Powered Analysis
- Automatic speech transcription using Wav2Vec2
- Compares actual speech vs target transcript
- Detects stutter events (repetitions, prolongations, blocks, interjections)
- Calculates severity metrics and confidence scores
- Extracts precise timestamps for each stutter event
- Background processing via Celery (non-blocking for users)
 
### 3. Report Generation & Visualization
- Session reports (single recording analysis)
- Key findings and results displayed with interactive charts
- Progress tracking over time with Chart.js visualizations
- Severity indicators with color-coded badges
- Detailed stutter event timeline


## 🔐 User Roles

### Patient
- Upload audio recordings via browser microphone or file upload
- View detailed personal analysis reports with charts
- Track progress with visual indicators
- Access analysis history
- View severity classifications


## 🔄 Application Workflow

1. **Patient Registration**: User creates account via Username without password
2. **Login**: Username without password authenticates with Django session-based authentication
3. **Dashboard**: Patient views overview with recent recordings and analysis summaries
4. **Audio Recording**: 
   - Patient navigates to recording page (deflaut page)
   - Grants microphone permission
   - Records speech sample using Web Audio API
   - Preview and validate recording
5. **Audio Upload**: 
   - Patient uploads recorded audio (or uploads pre-recorded file)
   - Client-side validation (file size, format)
   - Server-side validation and storage
   - Recording status set to "pending" and show loading spinner
6. **Queue Processing**: Audio file is queued for AI analysis (Celery task) show loading with text "Processing"
7. **AI Analysis** (Background Process): 
   - Wav2Vec2 model transcribes audio
   - Compares with target transcript
   - Detects stutter patterns and timestamps
   - Calculates severity and metrics
   - Updates recording status to "processing" → "completed"
8. **Result Storage**: Analysis results saved to PostgreSQL database
9. **View Results**: 
   - Patient receives result auto-refresh via AJAX
   - Views detailed analysis report with charts
   - Explores stutter events timeline
10. **Progress Tracking**: 
    - System aggregates metrics over time
    - Generates trend visualizations
    - Calculates improvement scores

## 📊 Key Metrics Tracked

- **Mismatch Percentage**: % of characters that differ from target
- **CTC Loss Score**: Lower = better speech clarity
- **Stutter Frequency**: Number of stutters per minute
- **Stutter Duration**: Total seconds of stuttering detected
- **Severity Classification**: None / Mild / Moderate / Severe
- **Confidence Score**: Model's confidence in analysis
- **Improvement Score**: Progress indicator over time

## 🗄️ Database Schema Overview

### Core Models (core/models.py)
- **`User`** (Django built-in) - Authentication and user management
- **`Patient`** - Patient profiles with personal info, baseline severity, therapist assignment

### Diagnosis Models (diagnosis/models.py)
- **`AudioRecording`** - Uploaded audio files with status tracking (pending/processing/completed/failed)
- **`AnalysisResult`** - AI analysis output with transcripts, metrics, severity classification
- **`StutterEvent`** - Individual stutter occurrences with precise timestamps and event types

### Reports Models (reports/models.py)
- **`Report`** - Generated analysis reports with key findings 
- **`ProgressTracking`** - Historical progress data aggregation (future expansion)

## 🚀 Development Setup

### Prerequisites
- Python 3.9+
- PostgreSQL 13+ 
- Redis 6+ (for Celery task queue)
- Modern web browser with Web Audio API support

### Backend Setup
```bash
# Navigate to project directory
cd slaq_project

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure database (edit settings.py if needed)

# Run migrations
python manage.py migrate

# Create superuser for admin access
python manage.py createsuperuser

# Collect static files (CSS/JS)
python manage.py collectstatic --noinput

# Run development server
python manage.py runserver
```

### Start Celery Worker (Separate Terminal)
```bash
# Windows
celery -A slaq_project worker --pool=solo --loglevel=info

# Linux/Mac
celery -A slaq_project worker --loglevel=info
```

### Start Redis Server (Separate Terminal)
```bash
# Windows (if installed via chocolatey or WSL)
redis-server

# Or use Docker
docker run -p 6379:6379 redis:latest
```

### Frontend Setup
Django templates are served directly by Django views - **no separate build process needed**.

Ensure static files are properly configured in `settings.py`:
```python
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Tailwind CSS can be included via CDN or compiled separately
```

### Access the Application
- **Main App**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin
- **API Endpoints**: http://localhost:8000/api/ (JSON responses for AJAX)

## 🔌 API Endpoints & Routes

### Authentication Routes
- `GET /` - Landing page (home.html)
- `GET /register/` - User registration form
- `POST /register/` - Process registration
- `GET /login/` - Login page
- `POST /login/` - Authenticate user (session-based)
- `GET /logout/` - Logout and redirect

### Patient Dashboard & Profile
- `GET /dashboard/` - Patient dashboard with overview and recent recordings
- `GET /profile/` - View/edit patient profile
- `POST /profile/update/` - Update patient information

### Audio Recording & Upload
- `GET /record/` - Audio recording interface page
- `POST /recordings/upload/` - Upload audio file (handles form + file)
- `GET /recordings/` - List all patient recordings with status
- `GET /recordings/<id>/` - View specific recording details
- `DELETE /recordings/<id>/delete/` - Delete recording

### Analysis & Results
- `GET /analyses/<id>/` - View detailed analysis results page
- `GET /api/analyses/<id>/status/` - Check analysis status (AJAX endpoint, returns JSON)
- `GET /api/analyses/<id>/data/` - Get full analysis data (AJAX endpoint, returns JSON)

### Reports & Progress
- `GET /reports/` - View all analysis reports
- `GET /reports/<id>/` - View specific report with charts
- `GET /progress/` - Progress tracking page with visualizations
- `GET /api/progress/data/` - Get progress data for charts (AJAX endpoint, returns JSON)

### Admin Routes (Django Admin)
- `GET /admin/` - Django admin panel (staff/superuser only)

## 🎨 Template Structure

### Base Templates
- `base.html` - Main base template with navbar, footer, Tailwind CSS
- `base_dashboard.html` - Dashboard layout with sidebar navigation

### Core Templates (core/templates/)
- `home.html` - Landing page with app introduction
- `register.html` - User registration form
- `login.html` - Login form
- `profile.html` - Patient profile view/edit

### Diagnosis Templates (diagnosis/templates/)
- `dashboard.html` - Patient dashboard overview
- `record.html` - Audio recording interface with Web Audio API
- `recordings_list.html` - List of all recordings with status badges
- `recording_detail.html` - Single recording detail view

### Reports Templates (reports/templates/)
- `analysis_detail.html` - Detailed analysis results with Chart.js visualizations
- `reports_list.html` - List of all reports
- `progress_tracking.html` - Progress charts and trend analysis

### Static Files Structure
```
static/
├── css/
│   ├── tailwind.css         # Tailwind compiled CSS
│   ├── custom.css           # Custom styles
│   └── charts.css           # Chart styling
├── js/
│   ├── audio-recorder.js    # Web Audio API recording logic
│   ├── analysis-charts.js   # Chart.js configurations
│   ├── ajax-handlers.js     # Fetch API for AJAX calls
│   └── main.js             # General app functionality
└── images/
    └── logos/              # App logos and icons
```

## 🔒 Security Considerations

- **Django Session-Based Authentication**: Secure session management with CSRF protection
- **CSRF Tokens**: All forms include CSRF tokens to prevent cross-site request forgery
- **File Upload Validation**: 
  - File size limits (e.g., max 10MB per recording)
  - File type validation (only audio formats: WAV, MP3, WebM)
  - Sanitized file names
- **User Isolation**: Patients can only access their own data via view-level permissions
- **Secure Password Storage**: Django's built-in PBKDF2 password hashing
- **XSS Protection**: Django auto-escapes template variables
- **SQL Injection Protection**: Django ORM prevents SQL injection
- **HTTPS in Production**: Force HTTPS with `SECURE_SSL_REDIRECT = True`
- **Login Required Decorators**: Protected views require authentication
- **Media File Access Control**: Serve media files through Django views (not direct access)

### Production Security Checklist
```python
# settings.py (production)
DEBUG = False
ALLOWED_HOSTS = ['*']
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
```

## 💻 Frontend Technologies Deep Dive

### Tailwind CSS Integration
- **Responsive Design**: Mobile-first approach with breakpoints
  - Mobile: Default (< 640px)
  - Tablet: `sm:` (640px+) and `md:` (768px+)
  - Desktop: `lg:` (1024px+) and `xl:` (1280px+)
- **Component Classes**: Reusable utility classes for buttons, cards, forms
- **Color Scheme**: Consistent color palette for severity levels
  - Green: No stuttering
  - Yellow: Mild severity
  - Orange: Moderate severity
  - Red: Severe stuttering
- **Color Theme**: for overall website 
    - Black 
    - White

### Web Audio API Implementation
```javascript
// Audio recording flow
1. Request microphone permission: navigator.mediaDevices.getUserMedia()
2. Create MediaRecorder instance
3. Capture audio chunks in real-time
4. Visualize waveform using AnalyserNode
5. Stop recording and create Blob
6. Upload via FormData with Fetch API
```

### Chart.js Visualizations
- **Line Charts**: Progress over time (mismatch %, stutter frequency)
- **Bar Charts**: Weekly/monthly aggregated metrics
- **Doughnut Charts**: Severity distribution
- **Timeline Charts**: Stutter events within recording

### AJAX with Fetch API
```javascript
// Polling for analysis status
async function checkAnalysisStatus(recordingId) {
    const response = await fetch(`/api/analyses/${recordingId}/status/`);
    const data = await response.json();
    if (data.status === 'completed') {
        // Update UI with results
        loadAnalysisData(recordingId);
    }
}
```

### Progressive Enhancement
- **Graceful Degradation**: Works without JavaScript (forms submit normally)
- **Enhanced with JS**: AJAX, real-time updates, smooth transitions
- **Accessibility**: ARIA labels, keyboard navigation, screen reader support

## 📱 Responsive Design Approach

### Mobile (< 640px)
- Vertical stack layout
- Full-width buttons and cards
- Simplified navigation (hamburger menu)
- Touch-optimized recording controls

### Tablet (640px - 1024px)
- Two-column grid for lists
- Sidebar navigation
- Larger touch targets

### Desktop (1024px+)
- Multi-column dashboard layout
- Sidebar with expanded menu
- Data tables with sorting/filtering
- Hover effects and tooltips

## 🔧 Development Tools & Workflow

### Code Organization
```
slaq_project/
├── core/              # User authentication, patient profiles
├── diagnosis/         # Audio upload, AI analysis
├── reports/           # Report generation
├── static/            # CSS, JS, images
├── templates/         # Base templates
└── slaq_project/      # Settings, URLs, WSGI
```

### Environment Variables (.env)
```bash
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379/0
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/audio-recording
# ... make changes ...
git add .
git commit -m "feat: implement Web Audio API recording"
git push origin feature/audio-recording
# Create pull request
```

### Deployment Checklist
- [ ] Set `DEBUG=False`
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Set secure cookies (HTTPS)
- [ ] Run `collectstatic`
- [ ] Run database migrations
- [ ] Configure Celery workers
- [ ] Set up Redis persistence
- [ ] Configure media file storage (S3/CloudFront)
- [ ] Set up SSL certificate
- [ ] Configure backup strategy
- [ ] Set up monitoring and logging

---

## 📚 Additional Resources

### Django Documentation
- [Django Official Docs](https://docs.djangoproject.com/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [Celery Documentation](https://docs.celeryproject.org/)

### Frontend Resources
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Web Audio API Guide](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [Chart.js Documentation](https://www.chartjs.org/docs/)

### AI/ML Resources
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Wav2Vec2 Model](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Speech Recognition Guide](https://huggingface.co/tasks/automatic-speech-recognition)

---

**Last Updated**: November 9, 2025  
**Version**: 1.0.0  
**Architecture**: Django Monolithic with Celery Workers

## 📈 Scalability Considerations

- **Celery for Async Processing**: Prevents blocking - AI analysis runs in background workers
- **Redis Queue**: Handles high volumes of concurrent analysis tasks
- **Media Files Organization**: Structured by patient/year/month for efficient storage
- **Database Indexes**: Optimized queries with indexes on frequently accessed fields
- **Static File Serving**: 
  - Use WhiteNoise for static files in production
  - Or CDN (CloudFront, CloudFlare) for global distribution
- **Horizontal Scaling**: 
  - Can add multiple Celery workers for parallel processing
  - Database connection pooling with PgBouncer
- **Caching Strategy**: Redis for session storage and result caching
- **Pagination**: Implement pagination for recordings/reports lists
- **Database Optimization**: 
  - Use `select_related()` and `prefetch_related()` to reduce queries
  - Regular vacuum and analyze on PostgreSQL

### Performance Monitoring
- Django Debug Toolbar (development)
- Logging with rotating file handlers
- Celery task monitoring with Flower
- Database query performance tracking

## 🧪 Testing Strategy

### Backend Testing
- **Unit Tests**: Test models, business logic, utility functions
  - Model methods and properties
  - Analysis calculations
  - Severity classification logic
- **View Tests**: Test Django views and responses
  - Authentication flows
  - Form validation
  - AJAX endpoint responses
- **Integration Tests**: Test complete workflows
  - File upload to analysis completion
  - Report generation pipeline
- **Celery Task Tests**: Mock AI model for faster testing

### Frontend Testing
- **Template Rendering**: Test template context and display
- **JavaScript Unit Tests**: Test audio recorder, AJAX handlers
- **Browser Testing**: Test across Chrome, Firefox, Safari
- **Responsive Design**: Test mobile, tablet, desktop layouts

### Test Data
- Sample audio files (various durations and qualities)
- Fixtures for users, patients, recordings
- Mock analysis results

### Running Tests
```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test core
python manage.py test diagnosis

# With coverage
coverage run --source='.' manage.py test
coverage report
```

## 🚧 Future Enhancements

### Phase 1 (Current MVP)
- ✅ Audio recording and upload
- ✅ AI-powered stutter detection
- ✅ Basic analysis reports
- ✅ Progress visualization

### Phase 2 (Next Steps)
- 🔄 Real-time audio analysis feedback during recording
- 🔄 Email notifications for completed analyses
- 🔄 PDF report generation and download
- 🔄 Advanced filtering and search for recordings
- 🔄 Therapy recommendations based on analysis patterns

### Phase 3 (Advanced Features)
- Mobile-responsive PWA (Progressive Web App)
- Multi-language support (Spanish, French, etc.)
- Advanced ML models for better accuracy (fine-tuned Wav2Vec2)
- Gamification for patient engagement (badges, streaks)
- Social features (anonymous progress sharing, support groups)

### Phase 4 (Enterprise)
- Therapist portal and patient management
- Telehealth integration (video consultations)
- Voice biometrics for user verification
- Multi-tenant architecture for clinics
- Insurance integration and billing
- Mobile apps (React Native or Flutter)
- Real-time collaborative sessions

### Technical Debt & Improvements
- Implement comprehensive error handling
- Add request rate limiting
- Optimize database queries with caching
- Implement webhook notifications
- Add admin dashboard with analytics
- Dockerize application for easier deployment
- CI/CD pipeline (GitHub Actions, GitLab CI)
- Automated backups and disaster recovery



