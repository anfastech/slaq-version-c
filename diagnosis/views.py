# diagnosis/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from django.conf import settings
from django.db.models import Avg, Count, Q

from .models import AudioRecording, AnalysisResult, StutterEvent
from .tasks import process_audio_recording
from core.models import Patient

import logging
import os

logger = logging.getLogger(__name__)


@login_required
def record_audio(request):
    """Display recording interface"""
    return render(request, 'diagnosis/record.html')


@login_required
def upload_recording(request):
    """Handle audio file upload"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        patient = request.user.patient_profile
        audio_file = request.FILES.get('audio_file')
        if not audio_file:
            return JsonResponse({'error': 'No audio file provided'}, status=400)
        
        if audio_file.size > settings.MAX_AUDIO_FILE_SIZE:
            max_size_mb = settings.MAX_AUDIO_FILE_SIZE / (1024*1024)
            return JsonResponse({'error': f'File too large. Max size: {max_size_mb}MB'}, status=400)
        
        file_ext = os.path.splitext(audio_file.name)[1].lower()
        if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
            return JsonResponse({'error': f'Invalid file format. Allowed: {", ".join(settings.ALLOWED_AUDIO_FORMATS)}'}, status=400)
        
        recording = AudioRecording.objects.create(
            patient=patient,
            audio_file=audio_file,
            file_size_bytes=audio_file.size,
            status='pending'
        )
        
        logger.info(f"Recording {recording.id} uploaded by {patient.user.username}")
        process_audio_recording.delay(recording.id)
        
        return JsonResponse({
            'success': True,
            'recording_id': recording.id,
            'message': 'Audio uploaded successfully. Processing started.'
        }, status=201)
        
    except Patient.DoesNotExist:
        return JsonResponse({'error': 'Patient profile not found'}, status=404)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def recordings_list(request):
    """Display list of patient's recordings"""
    try:
        patient = request.user.patient_profile
        status_filter = request.GET.get('status', None)
        recordings = AudioRecording.objects.filter(patient=patient).order_by('-recorded_at')
        
        if status_filter:
            recordings = recordings.filter(status=status_filter)
        
        total_count = recordings.count()
        completed_count = recordings.filter(status='completed').count()
        pending_count = recordings.filter(status='pending').count()
        processing_count = recordings.filter(status='processing').count()
        failed_count = recordings.filter(status='failed').count()
        
        context = {
            'recordings': recordings,
            'total_count': total_count,
            'completed_count': completed_count,
            'pending_count': pending_count,
            'processing_count': processing_count,
            'failed_count': failed_count,
            'status_filter': status_filter,
        }
        
        return render(request, 'diagnosis/recordings_list.html', context)
        
    except Patient.DoesNotExist:
        messages.error(request, 'Patient profile not found')
        return redirect('core:dashboard')


@login_required
def recording_detail(request, recording_id):
    """Display single recording details"""
    try:
        patient = request.user.patient_profile
        recording = get_object_or_404(AudioRecording, id=recording_id, patient=patient)
        analysis = None
        if recording.status == 'completed':
            try:
                analysis = recording.analysis
            except AnalysisResult.DoesNotExist:
                pass
        
        context = {
            'recording': recording,
            'analysis': analysis,
        }
        
        return render(request, 'diagnosis/recording_detail.html', context)
        
    except Patient.DoesNotExist:
        messages.error(request, 'Patient profile not found')
        return redirect('core:dashboard')


@login_required
def analysis_detail(request, analysis_id):
    """Display analysis results with detailed metrics"""
    try:
        patient = request.user.patient_profile
        analysis = get_object_or_404(AnalysisResult, id=analysis_id, recording__patient=patient)
        events = analysis.events.all().order_by('start_time')
        total_events = events.count()
        event_types = events.values('event_type').annotate(count=Count('event_type'))
        
        context = {
            'analysis': analysis,
            'recording': analysis.recording,
            'events': events,
            'total_events': total_events,
            'event_types': event_types,
        }
        
        return render(request, 'diagnosis/analysis_detail.html', context)
        
    except Patient.DoesNotExist:
        messages.error(request, 'Patient profile not found')
        return redirect('core:dashboard')


@login_required
def delete_recording(request, recording_id):
    """Delete a recording"""
    if request.method != 'POST':
        return redirect('diagnosis:recordings_list')
    
    try:
        patient = request.user.patient_profile
        recording = get_object_or_404(AudioRecording, id=recording_id, patient=patient)
        
        if recording.audio_file:
            recording.audio_file.delete()
        
        recording.delete()
        messages.success(request, 'Recording deleted successfully')
        
    except Patient.DoesNotExist:
        messages.error(request, 'Patient profile not found')
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        messages.error(request, 'Failed to delete recording')
    
    return redirect('diagnosis:recordings_list')


@login_required
def check_status(request, recording_id):
    """API endpoint to check recording status (for AJAX polling)"""
    try:
        patient = request.user.patient_profile
        recording = get_object_or_404(AudioRecording, id=recording_id, patient=patient)
        
        response_data = {
            'id': recording.id,
            'status': recording.status,
            'error_message': recording.error_message,
        }
        
        if recording.status == 'completed':
            try:
                analysis = recording.analysis
                response_data.update({
                    'analysis_id': analysis.id,
                    'severity': analysis.severity,
                    'mismatch_percentage': float(analysis.mismatch_percentage),
                })
            except AnalysisResult.DoesNotExist:
                pass
        
        return JsonResponse(response_data)
        
    except Patient.DoesNotExist:
        return JsonResponse({'error': 'Patient profile not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
