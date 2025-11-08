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
        
        # Calculate stats
        pending_count = AudioRecording.objects.filter(patient=patient, status='pending').count()
        processing_count = AudioRecording.objects.filter(patient=patient, status='processing').count()
        
        context = {
            'patient': patient,
            'recordings': recordings,
            'total_recordings': total_recordings,
            'completed_count': completed.count(),
            'pending_count': pending_count,
            'processing_count': processing_count,
            'latest_analysis': latest_analysis,
        }
        
        return render(request, 'core/dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading dashboard: {e}')
        return redirect('core:home')


@login_required
def profile(request):
    """View patient profile"""
    try:
        patient = request.user.patient_profile
        
        # Get total recordings and analyses
        total_recordings = AudioRecording.objects.filter(patient=patient).count()
        completed_analyses = AudioRecording.objects.filter(patient=patient, status='completed').count()
        
        context = {
            'patient': patient,
            'total_recordings': total_recordings,
            'completed_analyses': completed_analyses,
        }
        
        return render(request, 'core/profile.html', context)
    except Exception as e:
        messages.error(request, f'Error loading profile: {e}')
        return redirect('core:dashboard')
