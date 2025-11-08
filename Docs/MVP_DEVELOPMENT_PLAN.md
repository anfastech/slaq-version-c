# SLAQ MVP - Complete Development Plan

## 📋 Project Status Overview

### Current State Analysis
- ✅ Django project structure created
- ✅ Models defined (Patient, AudioRecording, AnalysisResult, etc.)
- ✅ Celery tasks implemented (AI processing)
- ✅ REST API views (needs conversion to template views)
- ❌ Templates not created
- ❌ Static files (CSS/JS) not created
- ❌ URLs not configured
- ❌ Authentication views not implemented
- ❌ Frontend interface not built

### What Needs to Be Done

## 🎯 PHASE 1: Foundation Setup (Week 1)

### 1.1 Update Settings for MVP
**File**: `slaq_project/settings.py`
- ✅ Already configured for PostgreSQL
- ✅ Celery configured
- ⚠️ Remove REST Framework dependencies (keeping session auth only)
- ⚠️ Update CORS settings (not needed for monolithic app)
- ⚠️ Configure templates directory
- ⚠️ Configure static files properly
- ⚠️ Add LOGIN_URL and REDIRECT settings

### 1.2 Create Base URL Configuration
**File**: `slaq_project/urls.py`
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),              # Auth & dashboard
    path('diagnosis/', include('diagnosis.urls')),  # Audio & analysis
    path('reports/', include('reports.urls')),     # Reports (future)
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
```

### 1.3 Create Templates Directory Structure
```
templates/
├── base.html                    # Main base template
├── components/
│   ├── navbar.html
│   ├── footer.html
│   ├── messages.html
│   └── severity_badge.html
├── core/
│   ├── home.html               # Landing page
│   ├── register.html           # Registration
│   ├── login.html              # Login
│   ├── dashboard.html          # Patient dashboard
│   └── profile.html            # Profile view
└── diagnosis/
    ├── record.html             # Audio recording page
    ├── recordings_list.html    # All recordings
    ├── recording_detail.html   # Single recording
    └── analysis_detail.html    # Analysis results
```

### 1.4 Create Static Files Structure
```
static/
├── css/
│   ├── main.css               # Main custom styles
│   └── components.css         # Reusable component styles
├── js/
│   ├── main.js                # Global JS utilities
│   ├── audio-recorder.js      # Web Audio API recording
│   ├── analysis-charts.js     # Chart.js visualizations
│   └── ajax-handler.js        # AJAX status checks
└── images/
    ├── logo.png
    └── icons/
        ├── microphone.svg
        ├── upload.svg
        └── analysis.svg
```

## 🎯 PHASE 2: Authentication & Core Views (Days 1-3)

### 2.1 Core URLs Configuration
**File**: `core/urls.py`
```python
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'core'

urlpatterns = [
    # Landing & Auth
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='core/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='core:home'), name='logout'),
    
    # Dashboard & Profile
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
]
```

### 2.2 Core Views Implementation
**File**: `core/views.py`
```python
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib import messages
from .forms import PatientRegistrationForm
from diagnosis.models import AudioRecording, AnalysisResult

def home(request):
    """Landing page"""
    if request.user.is_authenticated:
        return redirect('core:dashboard')
    return render(request, 'core/home.html')

def register(request):
    """Patient registration"""
    if request.method == 'POST':
        form = PatientRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Welcome to SLAQ! Your account has been created.')
            return redirect('core:dashboard')
    else:
        form = PatientRegistrationForm()
    
    return render(request, 'core/register.html', {'form': form})

@login_required
def dashboard(request):
    """Patient dashboard"""
    try:
        patient = request.user.patient_profile
        
        # Get recordings
        recordings = AudioRecording.objects.filter(patient=patient).order_by('-recorded_at')[:5]
        total_recordings = AudioRecording.objects.filter(patient=patient).count()
        
        # Get completed analyses
        completed = recordings.filter(status='completed')
        latest_analysis = None
        if completed.exists():
            latest_recording = completed.first()
            latest_analysis = getattr(latest_recording, 'analysis', None)
        
        context = {
            'patient': patient,
            'recordings': recordings,
            'total_recordings': total_recordings,
            'completed_count': completed.count(),
            'latest_analysis': latest_analysis,
        }
        
        return render(request, 'core/dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading dashboard: {e}')
        return redirect('core:home')

@login_required
def profile(request):
    """View patient profile"""
    patient = request.user.patient_profile
    return render(request, 'core/profile.html', {'patient': patient})
```

### 2.3 Registration Form
**File**: `core/forms.py`
```python
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Patient
from datetime import date

class PatientRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    date_of_birth = forms.DateField(
        required=True,
        widget=forms.DateInput(attrs={'type': 'date'})
    )
    phone_number = forms.CharField(max_length=15, required=False)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 
                  'password1', 'password2']
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Email already registered')
        return email
    
    def clean_date_of_birth(self):
        dob = self.cleaned_data.get('date_of_birth')
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if age < 5:
            raise forms.ValidationError('Must be at least 5 years old')
        if age > 120:
            raise forms.ValidationError('Invalid date of birth')
        return dob
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
            # Create patient profile
            Patient.objects.create(
                user=user,
                date_of_birth=self.cleaned_data['date_of_birth'],
                phone_number=self.cleaned_data.get('phone_number', '')
            )
        
        return user
```

## 🎯 PHASE 3: Audio Recording Interface (Days 4-6)

### 3.1 Diagnosis URLs
**File**: `diagnosis/urls.py`
```python
from django.urls import path
from . import views

app_name = 'diagnosis'

urlpatterns = [
    # Recording
    path('record/', views.record_audio, name='record'),
    path('recordings/', views.recordings_list, name='recordings_list'),
    path('recordings/<int:pk>/', views.recording_detail, name='recording_detail'),
    path('recordings/<int:pk>/delete/', views.delete_recording, name='delete_recording'),
    path('recordings/upload/', views.upload_recording, name='upload_recording'),
    
    # Analysis
    path('analysis/<int:pk>/', views.analysis_detail, name='analysis_detail'),
    
    # AJAX endpoints
    path('api/status/<int:recording_id>/', views.check_status, name='check_status'),
]
```

### 3.2 Diagnosis Views (Template-based)
**File**: `diagnosis/views.py` (REPLACE existing REST views)
```python
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_POST
import os

from .models import AudioRecording, AnalysisResult
from .tasks import process_audio_recording
from .forms import AudioUploadForm

@login_required
def record_audio(request):
    """Audio recording interface"""
    return render(request, 'diagnosis/record.html')

@login_required
def recordings_list(request):
    """List all patient recordings"""
    patient = request.user.patient_profile
    recordings = AudioRecording.objects.filter(patient=patient).order_by('-recorded_at')
    
    # Filter by status if requested
    status_filter = request.GET.get('status')
    if status_filter:
        recordings = recordings.filter(status=status_filter)
    
    context = {
        'recordings': recordings,
        'status_filter': status_filter,
    }
    
    return render(request, 'diagnosis/recordings_list.html', context)

@login_required
def recording_detail(request, pk):
    """View single recording details"""
    patient = request.user.patient_profile
    recording = get_object_or_404(AudioRecording, pk=pk, patient=patient)
    
    # Get analysis if available
    analysis = None
    if recording.status == 'completed':
        analysis = getattr(recording, 'analysis', None)
    
    context = {
        'recording': recording,
        'analysis': analysis,
    }
    
    return render(request, 'diagnosis/recording_detail.html', context)

@login_required
def analysis_detail(request, pk):
    """View detailed analysis results"""
    patient = request.user.patient_profile
    analysis = get_object_or_404(AnalysisResult, pk=pk, recording__patient=patient)
    
    # Get stutter events
    events = analysis.events.all().order_by('start_time')
    
    context = {
        'analysis': analysis,
        'recording': analysis.recording,
        'events': events,
    }
    
    return render(request, 'diagnosis/analysis_detail.html', context)

@login_required
@require_POST
def upload_recording(request):
    """Handle audio file upload"""
    form = AudioUploadForm(request.POST, request.FILES)
    
    if form.is_valid():
        try:
            patient = request.user.patient_profile
            audio_file = request.FILES['audio_file']
            
            # Validate file size
            if audio_file.size > settings.MAX_AUDIO_FILE_SIZE:
                messages.error(request, 'File too large. Maximum size is 10MB.')
                return redirect('diagnosis:record')
            
            # Validate file extension
            file_ext = os.path.splitext(audio_file.name)[1].lower()
            if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
                messages.error(request, f'Invalid format. Allowed: {", ".join(settings.ALLOWED_AUDIO_FORMATS)}')
                return redirect('diagnosis:record')
            
            # Create recording
            recording = AudioRecording.objects.create(
                patient=patient,
                audio_file=audio_file,
                file_size_bytes=audio_file.size,
                status='pending'
            )
            
            # Queue for processing
            process_audio_recording.delay(recording.id)
            
            messages.success(request, 'Audio uploaded successfully! Analysis in progress...')
            return redirect('diagnosis:recording_detail', pk=recording.id)
            
        except Exception as e:
            messages.error(request, f'Upload failed: {str(e)}')
            return redirect('diagnosis:record')
    
    messages.error(request, 'Invalid form submission')
    return redirect('diagnosis:record')

@login_required
@require_POST
def delete_recording(request, pk):
    """Delete a recording"""
    patient = request.user.patient_profile
    recording = get_object_or_404(AudioRecording, pk=pk, patient=patient)
    
    recording.delete()
    messages.success(request, 'Recording deleted successfully')
    return redirect('diagnosis:recordings_list')

@login_required
def check_status(request, recording_id):
    """AJAX endpoint to check recording status"""
    patient = request.user.patient_profile
    recording = get_object_or_404(AudioRecording, pk=recording_id, patient=patient)
    
    data = {
        'status': recording.status,
        'has_analysis': hasattr(recording, 'analysis'),
    }
    
    if recording.status == 'completed' and hasattr(recording, 'analysis'):
        analysis = recording.analysis
        data['analysis'] = {
            'id': analysis.id,
            'severity': analysis.severity,
            'mismatch_percentage': analysis.mismatch_percentage,
        }
    
    return JsonResponse(data)
```

### 3.3 Audio Upload Form
**File**: `diagnosis/forms.py`
```python
from django import forms

class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(
        required=True,
        widget=forms.FileInput(attrs={
            'accept': 'audio/*',
            'class': 'hidden'
        })
    )
```

## 🎯 PHASE 4: Templates Implementation (Days 7-10)

### 4.1 Base Template
**File**: `templates/base.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SLAQ - Stuttering Analysis{% endblock %}</title>
    
    <!-- Tailwind CSS CDN (MVP) -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Custom Tailwind Config -->
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'brand-green': '#009050',
                        'severity-none': '#10b981',
                        'severity-mild': '#fbbf24',
                        'severity-moderate': '#f97316',
                        'severity-severe': '#ef4444',
                    }
                }
            }
        }
    </script>
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{% static 'css/main.css' %}">
    
    {% block extra_css %}{% endblock %}
</head>
<body class="bg-gray-50 min-h-screen">
    
    {% include 'components/navbar.html' %}
    
    <!-- Messages -->
    {% include 'components/messages.html' %}
    
    <!-- Main Content -->
    <main class="container mx-auto px-4 py-8">
        {% block content %}{% endblock %}
    </main>
    
    {% include 'components/footer.html' %}
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    
    <!-- Custom JS -->
    <script src="{% static 'js/main.js' %}"></script>
    
    {% block extra_js %}{% endblock %}
</body>
</html>
```

[Continue with all other templates...]

## 🎯 PHASE 5: Static Files (Days 11-12)

### 5.1 Audio Recorder JavaScript
**File**: `static/js/audio-recorder.js`
```javascript
class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.audioContext = null;
        this.analyser = null;
    }
    
    async initialize() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(this.stream);
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.setupVisualizer();
            return true;
        } catch (error) {
            console.error('Microphone access denied:', error);
            return false;
        }
    }
    
    setupVisualizer() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        const source = this.audioContext.createMediaStreamSource(this.stream);
        source.connect(this.analyser);
        this.analyser.fftSize = 2048;
    }
    
    start() {
        this.audioChunks = [];
        this.mediaRecorder.start();
    }
    
    stop() {
        return new Promise((resolve) => {
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                resolve(audioBlob);
            };
            this.mediaRecorder.stop();
        });
    }
    
    visualize(canvas) {
        const canvasContext = canvas.getContext('2d');
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            requestAnimationFrame(draw);
            
            this.analyser.getByteTimeDomainData(dataArray);
            
            canvasContext.fillStyle = 'rgb(255, 255, 255)';
            canvasContext.fillRect(0, 0, canvas.width, canvas.height);
            
            canvasContext.lineWidth = 2;
            canvasContext.strokeStyle = 'rgb(0, 144, 80)';
            canvasContext.beginPath();
            
            const sliceWidth = canvas.width / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * canvas.height / 2;
                
                if (i === 0) {
                    canvasContext.moveTo(x, y);
                } else {
                    canvasContext.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            canvasContext.lineTo(canvas.width, canvas.height / 2);
            canvasContext.stroke();
        };
        
        draw();
    }
    
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}
```

## 🎯 PHASE 6: Testing & Polish (Days 13-14)

### Testing Checklist
- [ ] User registration works
- [ ] Login/logout works
- [ ] Dashboard displays correctly
- [ ] Audio recording works in browser
- [ ] File upload works
- [ ] Celery processes audio
- [ ] Analysis displays correctly
- [ ] Charts render properly
- [ ] Responsive on mobile
- [ ] All links work

## 📦 File Creation Checklist

### Configuration Files
- [x] settings.py (update needed)
- [ ] urls.py (main) - CREATE
- [ ] celery.py - EXISTS
- [ ] .env.example - CREATE

### Core App
- [ ] core/urls.py - CREATE
- [ ] core/views.py - CREATE
- [ ] core/forms.py - CREATE

### Diagnosis App
- [ ] diagnosis/urls.py - CREATE (empty, needs content)
- [ ] diagnosis/views.py - REPLACE (convert from REST to templates)
- [ ] diagnosis/forms.py - CREATE

### Templates (ALL NEW)
- [ ] templates/base.html
- [ ] templates/components/navbar.html
- [ ] templates/components/footer.html
- [ ] templates/components/messages.html
- [ ] templates/components/severity_badge.html
- [ ] templates/core/home.html
- [ ] templates/core/register.html
- [ ] templates/core/login.html
- [ ] templates/core/dashboard.html
- [ ] templates/core/profile.html
- [ ] templates/diagnosis/record.html
- [ ] templates/diagnosis/recordings_list.html
- [ ] templates/diagnosis/recording_detail.html
- [ ] templates/diagnosis/analysis_detail.html

### Static Files (ALL NEW)
- [ ] static/css/main.css
- [ ] static/css/components.css
- [ ] static/js/main.js
- [ ] static/js/audio-recorder.js
- [ ] static/js/analysis-charts.js
- [ ] static/js/ajax-handler.js
- [ ] static/images/logo.png

## 🚀 Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir templates static\css static\js static\images logs

# Setup database
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser

# Create static directories
python manage.py collectstatic --noinput
```

### Run Development
```bash
# Terminal 1: Django
python manage.py runserver

# Terminal 2: Celery
celery -A slaq_project worker --pool=solo -l info

# Terminal 3: Redis
redis-server
```

## 📈 Progress Tracking

### Week 1: Foundation
- [ ] Day 1: Settings, URLs, directory structure
- [ ] Day 2: Core views and forms
- [ ] Day 3: Authentication templates

### Week 2: Recording Interface
- [ ] Day 4: Recording page and upload
- [ ] Day 5: Web Audio API integration
- [ ] Day 6: Recordings list and detail pages

### Week 3: Analysis Display
- [ ] Day 7: Analysis detail page
- [ ] Day 8: Chart.js integration
- [ ] Day 9: Dashboard page
- [ ] Day 10: Profile page

### Week 4: Polish
- [ ] Day 11-12: Responsive design
- [ ] Day 13: Testing
- [ ] Day 14: Bug fixes and deployment prep

---

**Next Steps**: Start with Phase 1 - Update settings and create URL configurations.
