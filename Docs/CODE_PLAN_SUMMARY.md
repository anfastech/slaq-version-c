# 🚀 SLAQ MVP - Complete Code Plan & Implementation Guide

## 📊 Current Status

### ✅ What's Already Done
1. **Project Structure** - Django apps created (core, diagnosis, reports)
2. **Models** - All database models defined
3. **Celery Tasks** - AI processing tasks implemented
4. **AI Engine** - Stuttering detection logic exists
5. **Settings** - PostgreSQL, Celery, Redis configured
6. **Requirements** - All dependencies listed

### ⚠️ What Needs to Be Built
1. **URL Configuration** - ✅ CREATED
2. **Template Views** - ✅ CREATED (core views)
3. **Forms** - ✅ CREATED (registration form)
4. **Templates** - ❌ NOT CREATED (14 HTML files needed)
5. **Static Files** - ❌ NOT CREATED (CSS, JS files needed)
6. **Diagnosis Views** - ❌ NEEDS CONVERSION (from REST to templates)
7. **Reports Views** - ❌ NOT CREATED

---

## 📂 Complete File Structure

```
slaq_project/
├── manage.py                          ✅ EXISTS
├── requirements.txt                   ✅ EXISTS
├── .env                              ✅ EXISTS
├── README.md                         ✅ EXISTS
│
├── slaq_project/                     # Main project settings
│   ├── __init__.py                   ✅ EXISTS
│   ├── settings.py                   ✅ EXISTS (needs minor updates)
│   ├── urls.py                       ✅ CREATED
│   ├── wsgi.py                       ✅ EXISTS
│   ├── asgi.py                       ✅ EXISTS
│   └── celery.py                     ✅ EXISTS
│
├── core/                             # User authentication & dashboard
│   ├── __init__.py                   ✅ EXISTS
│   ├── admin.py                      ✅ EXISTS
│   ├── apps.py                       ✅ EXISTS
│   ├── models.py                     ✅ EXISTS
│   ├── views.py                      ✅ CREATED (home, register, dashboard, profile)
│   ├── urls.py                       ✅ CREATED
│   ├── forms.py                      ✅ CREATED (PatientRegistrationForm)
│   ├── serializers.py                ✅ EXISTS (can delete for MVP)
│   └── migrations/                   ✅ EXISTS
│
├── diagnosis/                        # Audio recording & analysis
│   ├── __init__.py                   ✅ EXISTS
│   ├── admin.py                      ✅ EXISTS
│   ├── apps.py                       ✅ EXISTS
│   ├── models.py                     ✅ EXISTS
│   ├── views.py                      ⚠️ EXISTS (needs replacement with template views)
│   ├── urls.py                       ❌ EMPTY (needs content)
│   ├── forms.py                      ❌ NEEDS CREATION
│   ├── serializers.py                ✅ EXISTS (can delete for MVP)
│   ├── tasks.py                      ✅ EXISTS
│   ├── ai_engine/                    ✅ EXISTS
│   │   ├── __init__.py
│   │   ├── detect_stuttering.py
│   │   ├── model_loader.py
│   │   └── utils.py
│   └── migrations/                   ✅ EXISTS
│
├── reports/                          # Reports (minimal for MVP)
│   ├── __init__.py                   ✅ EXISTS
│   ├── admin.py                      ✅ EXISTS
│   ├── apps.py                       ✅ EXISTS
│   ├── models.py                     ✅ EXISTS
│   ├── views.py                      ❌ EMPTY
│   ├── urls.py                       ❌ EMPTY
│   ├── serializers.py                ✅ EXISTS (can delete for MVP)
│   └── migrations/                   ✅ EXISTS
│
├── templates/                        # All HTML templates
│   ├── base.html                     ❌ CREATE
│   ├── components/
│   │   ├── navbar.html               ❌ CREATE
│   │   ├── footer.html               ❌ CREATE
│   │   ├── messages.html             ❌ CREATE
│   │   └── severity_badge.html       ❌ CREATE
│   ├── core/
│   │   ├── home.html                 ❌ CREATE
│   │   ├── register.html             ❌ CREATE
│   │   ├── login.html                ❌ CREATE
│   │   ├── dashboard.html            ❌ CREATE
│   │   └── profile.html              ❌ CREATE
│   └── diagnosis/
│       ├── record.html               ❌ CREATE
│       ├── recordings_list.html      ❌ CREATE
│       ├── recording_detail.html     ❌ CREATE
│       └── analysis_detail.html      ❌ CREATE
│
├── static/                           # CSS, JavaScript, Images
│   ├── css/
│   │   ├── main.css                  ❌ CREATE
│   │   └── components.css            ❌ CREATE
│   ├── js/
│   │   ├── main.js                   ❌ CREATE
│   │   ├── audio-recorder.js         ❌ CREATE
│   │   ├── analysis-charts.js        ❌ CREATE
│   │   └── ajax-handler.js           ❌ CREATE
│   └── images/
│       ├── logo.png                  ❌ CREATE
│       └── icons/                    ❌ CREATE
│
├── media/                            # User uploads
│   └── recordings/                   ✅ EXISTS
│
├── ml_models/                        # AI models
│   └── wav2vec2/                     ✅ EXISTS
│
└── Docs/                             # Documentation
    ├── context.md                    ✅ EXISTS
    ├── mvp_context.md                ✅ EXISTS
    ├── MVP_DEVELOPMENT_PLAN.md       ✅ CREATED
    └── CODE_PLAN.md                  ✅ THIS FILE
```

---

## 🔄 Conversion Strategy: REST API → Template Views

### Current State (REST API)
Your `diagnosis/views.py` is currently built for REST API with:
- ViewSets (AudioRecordingViewSet, AnalysisResultViewSet)
- Serializers
- JSON responses
- JWT authentication

### Target State (Template Views)
Need to convert to Django template views:
- Function-based views
- HTML responses
- Session authentication
- Form handling

### Example Conversion

**BEFORE (REST API):**
```python
class AudioRecordingViewSet(viewsets.ModelViewSet):
    serializer_class = AudioRecordingSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return AudioRecording.objects.filter(patient=self.request.user.patient_profile)
```

**AFTER (Template View):**
```python
@login_required
def recordings_list(request):
    patient = request.user.patient_profile
    recordings = AudioRecording.objects.filter(patient=patient).order_by('-recorded_at')
    return render(request, 'diagnosis/recordings_list.html', {'recordings': recordings})
```

---

## 🎯 Implementation Priority Order

### Phase 1: Core Foundation (Days 1-2)
**Goal**: Get authentication working

1. **Update Settings** (Minor)
   - Add `LOGIN_URL = '/login/'`
   - Add `LOGIN_REDIRECT_URL = '/dashboard/'`
   - Add `LOGOUT_REDIRECT_URL = '/'`

2. **Create Templates Directory**
   ```bash
   mkdir templates
   mkdir templates\components
   mkdir templates\core
   mkdir templates\diagnosis
   ```

3. **Create Base Template** (`templates/base.html`)
   - Tailwind CSS via CDN
   - Navbar, footer, messages
   - Block structure

4. **Create Auth Templates**
   - `templates/core/home.html` - Landing page
   - `templates/core/register.html` - Registration form
   - `templates/core/login.html` - Login form

5. **Test Authentication**
   ```bash
   python manage.py runserver
   # Test: http://localhost:8000/
   # Test: http://localhost:8000/register/
   # Test: http://localhost:8000/login/
   ```

### Phase 2: Dashboard (Day 3)
**Goal**: Show user data after login

1. **Create Dashboard Template** (`templates/core/dashboard.html`)
   - Welcome message
   - Stats cards (total recordings, completed analyses)
   - Recent recordings table
   - "Record New Audio" button

2. **Create Profile Template** (`templates/core/profile.html`)
   - User info display
   - Patient details
   - Stats

3. **Test Dashboard**
   - Login and verify dashboard displays
   - Check profile page

### Phase 3: Audio Recording (Days 4-6)
**Goal**: Record and upload audio

1. **Create Diagnosis Views** (`diagnosis/views.py`)
   - `record_audio()` - Recording interface
   - `upload_recording()` - Handle file upload
   - `recordings_list()` - List all recordings
   - `recording_detail()` - Single recording
   - `delete_recording()` - Delete recording
   - `check_status()` - AJAX status check

2. **Create Diagnosis URLs** (`diagnosis/urls.py`)

3. **Create Diagnosis Forms** (`diagnosis/forms.py`)
   - `AudioUploadForm`

4. **Create Record Template** (`templates/diagnosis/record.html`)
   - Microphone button
   - Record/Stop/Play controls
   - Waveform visualization (canvas)
   - File upload alternative

5. **Create Audio Recorder JS** (`static/js/audio-recorder.js`)
   - Web Audio API
   - MediaRecorder
   - Waveform visualization
   - Blob creation
   - FormData upload

6. **Create Recordings List Template** (`templates/diagnosis/recordings_list.html`)
   - Table/cards of recordings
   - Status badges
   - Filter options
   - Actions (view, delete)

7. **Test Recording Flow**
   - Record audio via microphone
   - Upload pre-recorded file
   - View in recordings list
   - Check Celery processes it

### Phase 4: Analysis Display (Days 7-9)
**Goal**: Show AI analysis results

1. **Create Analysis Detail View** (`diagnosis/views.py`)
   - `analysis_detail()` - Show results

2. **Create Analysis Template** (`templates/diagnosis/analysis_detail.html`)
   - Severity badge
   - Key metrics (mismatch %, CTC loss, confidence)
   - Transcript comparison
   - Mismatched characters list
   - Chart (Chart.js)

3. **Create Charts JS** (`static/js/analysis-charts.js`)
   - Bar chart for metrics
   - Initialize Chart.js

4. **Create Recording Detail Template** (`templates/diagnosis/recording_detail.html`)
   - Recording info
   - Link to analysis (if completed)
   - Status display

5. **Test Analysis Display**
   - Process a recording
   - View analysis results
   - Check charts render

### Phase 5: Polish & Testing (Days 10-14)
**Goal**: Make it production-ready

1. **Create Components**
   - `templates/components/navbar.html`
   - `templates/components/footer.html`
   - `templates/components/messages.html`
   - `templates/components/severity_badge.html`

2. **Create Custom CSS** (`static/css/main.css`)
   - Custom styles
   - Responsive utilities
   - Animations

3. **Create Utilities JS** (`static/js/main.js`)
   - Common functions
   - Form validation
   - Toast notifications

4. **AJAX Handler** (`static/js/ajax-handler.js`)
   - Poll for status updates
   - Update UI dynamically

5. **Responsive Testing**
   - Test on mobile (320px+)
   - Test on tablet (768px+)
   - Test on desktop (1024px+)

6. **Browser Testing**
   - Chrome
   - Firefox
   - Safari
   - Edge

7. **Functional Testing**
   - Complete user journey
   - Edge cases
   - Error handling

---

## 📝 Detailed Code Templates

### Template: Base HTML
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SLAQ - Stuttering Analysis{% endblock %}</title>
    
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'brand-green': '#009050',
                    }
                }
            }
        }
    </script>
    
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/main.css' %}">
    
    {% block extra_css %}{% endblock %}
</head>
<body class="bg-gray-50 min-h-screen flex flex-col">
    
    {% include 'components/navbar.html' %}
    
    {% include 'components/messages.html' %}
    
    <main class="flex-grow container mx-auto px-4 py-8">
        {% block content %}{% endblock %}
    </main>
    
    {% include 'components/footer.html' %}
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <script src="{% static 'js/main.js' %}"></script>
    
    {% block extra_js %}{% endblock %}
</body>
</html>
```

### Template: Navbar Component
```html
<nav class="bg-white shadow-lg">
    <div class="container mx-auto px-4">
        <div class="flex justify-between items-center py-4">
            <div class="flex items-center space-x-2">
                <div class="w-10 h-10 bg-brand-green rounded-full"></div>
                <span class="text-2xl font-bold text-brand-green">SLAQ</span>
            </div>
            
            <div class="hidden md:flex space-x-6">
                {% if user.is_authenticated %}
                    <a href="{% url 'core:dashboard' %}" class="text-gray-700 hover:text-brand-green">Dashboard</a>
                    <a href="{% url 'diagnosis:record' %}" class="text-gray-700 hover:text-brand-green">Record</a>
                    <a href="{% url 'diagnosis:recordings_list' %}" class="text-gray-700 hover:text-brand-green">Recordings</a>
                    <a href="{% url 'core:profile' %}" class="text-gray-700 hover:text-brand-green">Profile</a>
                    <a href="{% url 'core:logout' %}" class="text-red-600 hover:text-red-700">Logout</a>
                {% else %}
                    <a href="{% url 'core:home' %}" class="text-gray-700 hover:text-brand-green">Home</a>
                    <a href="{% url 'core:login' %}" class="text-gray-700 hover:text-brand-green">Login</a>
                    <a href="{% url 'core:register' %}" class="bg-brand-green text-white px-4 py-2 rounded-lg hover:bg-green-600">Get Started</a>
                {% endif %}
            </div>
        </div>
    </div>
</nav>
```

---

## 🧪 Testing Checklist

### Authentication
- [ ] User can access landing page
- [ ] User can register with valid data
- [ ] Registration validates email uniqueness
- [ ] Registration validates date of birth (age 5-120)
- [ ] User is auto-logged in after registration
- [ ] User can login with username/password
- [ ] User can logout
- [ ] Unauthenticated users redirected to login for protected pages

### Dashboard
- [ ] Dashboard displays after login
- [ ] Dashboard shows patient name
- [ ] Dashboard shows total recordings count
- [ ] Dashboard shows completed analyses count
- [ ] Dashboard shows recent recordings (max 5)
- [ ] Dashboard shows latest analysis summary
- [ ] "Record New Audio" button works

### Audio Recording
- [ ] Record page loads
- [ ] Microphone permission request appears
- [ ] Record button starts recording
- [ ] Waveform visualizes audio input
- [ ] Stop button stops recording
- [ ] Play button previews recording
- [ ] Upload button sends file to server
- [ ] File upload alternative works
- [ ] File size validation works (10MB limit)
- [ ] File format validation works (WAV, MP3, WebM, OGG)
- [ ] Celery task queues successfully
- [ ] Recording appears in list with "pending" status

### Recordings List
- [ ] List displays all user recordings
- [ ] List ordered by newest first
- [ ] Status badges display correctly (pending/processing/completed/failed)
- [ ] Filter by status works
- [ ] Click recording opens detail page
- [ ] Delete button removes recording

### Analysis Display
- [ ] Analysis page shows when recording completed
- [ ] Severity badge displays with correct color
- [ ] Mismatch percentage displays
- [ ] CTC loss score displays
- [ ] Confidence score displays
- [ ] Actual transcript displays
- [ ] Target transcript displays
- [ ] Mismatched characters list displays
- [ ] Chart renders correctly
- [ ] Recording info displays (date, duration)

### Responsive Design
- [ ] Works on mobile (320px)
- [ ] Works on mobile (375px - iPhone SE)
- [ ] Works on tablet (768px)
- [ ] Works on desktop (1024px)
- [ ] Works on large desktop (1440px)
- [ ] Navbar collapses on mobile
- [ ] Forms are usable on mobile
- [ ] Tables/cards adapt to screen size

### Browser Compatibility
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)
- [ ] Mobile Safari (iOS)
- [ ] Mobile Chrome (Android)

---

## 🚀 Quick Start Commands

### Initial Setup
```bash
# Activate virtual environment
venv\Scripts\activate

# Create necessary directories
mkdir templates templates\components templates\core templates\diagnosis
mkdir static\css static\js static\images static\images\icons

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput
```

### Development (3 Terminals)
```bash
# Terminal 1: Django
python manage.py runserver

# Terminal 2: Celery
celery -A slaq_project worker --pool=solo -l info

# Terminal 3: Redis
redis-server
```

### Database Access
```bash
# Django shell
python manage.py shell

# PostgreSQL
psql -U postgres -d slaq_db
```

---

## 📊 Progress Tracking

### Week 1: Foundation ✅ DONE
- [x] Settings configuration
- [x] URL routing
- [x] Core views
- [x] Core forms
- [ ] Base templates
- [ ] Authentication templates

### Week 2: Recording (NEXT)
- [ ] Diagnosis URLs
- [ ] Diagnosis views
- [ ] Diagnosis forms
- [ ] Recording templates
- [ ] Audio recorder JavaScript
- [ ] Recordings list template

### Week 3: Analysis
- [ ] Analysis views
- [ ] Analysis templates
- [ ] Charts JavaScript
- [ ] AJAX handlers
- [ ] Status polling

### Week 4: Polish
- [ ] Components
- [ ] Custom CSS
- [ ] Responsive design
- [ ] Testing
- [ ] Bug fixes
- [ ] Documentation

---

## 🎯 Next Steps

1. **Run migrations** to create database tables
2. **Create templates directory** and subdirectories
3. **Start with base.html** template
4. **Create authentication templates** (home, register, login)
5. **Test authentication flow**
6. **Move to dashboard** templates
7. **Then audio recording** interface

**Estimated Time to MVP**: 2-3 weeks of focused development

**Current Completion**: ~35% (backend done, frontend needed)

---

**Last Updated**: November 9, 2025  
**Next Priority**: Create templates directory and base.html
