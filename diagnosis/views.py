# diagnosis/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
import os
import logging

from .models import AudioRecording, AnalysisResult
from .tasks import process_audio_recording
from .forms import AudioUploadForm

logger = logging.getLogger(__name__)

@login_required
def record_audio(request):
    """Audio recording interface"""
    return render(request, 'diagnosis/record.html')

@login_required
def recordings_list(request):
    """List all patient recordings"""
    try:
        patient = request.user.patient_profile
        recordings = AudioRecording.objects.filter(patient=patient).order_by('-recorded_at')
        
        status_filter = request.GET.get('status')
        if status_filter:
            recordings = recordings.filter(status=status_filter)
        
        context = {
            'recordings': recordings,
            'status_filter': status_filter,
            'total_count': recordings.count(),
            'completed_count': recordings.filter(status='completed').count(),
            'pending_count': recordings.filter(status='pending').count(),
            'processing_count': recordings.filter(status='processing').count(),
            'failed_count': recordings.filter(status='failed').count(),
        }
        return render(request, 'diagnosis/recordings_list.html', context)
    except Exception as e:
        messages.error(request, f"Error loading recordings: {e}")
        return redirect('core:dashboard')

@login_required
def recording_detail(request, recording_id):
    """View single recording details"""
    try:
        patient = request.user.patient_profile
        recording = get_object_or_404(AudioRecording, id=recording_id, patient=patient)
        
        analysis = None
        if recording.status == 'completed':
            try:
                analysis = recording.analysis
            except AnalysisResult.DoesNotExist:
                pass
        
        return render(request, 'diagnosis/recording_detail.html', {
            'recording': recording, 
            'analysis': analysis
        })
    except Exception as e:
        messages.error(request, "Recording not found.")
        return redirect('diagnosis:recordings_list')

@login_required
def analysis_detail(request, analysis_id):
    """View detailed analysis results"""
    try:
        patient = request.user.patient_profile
        analysis = get_object_or_404(AnalysisResult, id=analysis_id, recording__patient=patient)
        
        # Process stutter events for the template
        events = []
        raw_events = analysis.stutter_timestamps or []
        
        for evt in raw_events:
            # Handle New Format (Dict)
            if isinstance(evt, dict):
                events.append({
                    'event_type': evt.get('type', 'dysfluency'),
                    'start_time': evt.get('start', 0),
                    'end_time': evt.get('end', 0),
                    'duration': evt.get('duration', 0),
                    'confidence': evt.get('confidence', 0.0)
                })
            # Handle Legacy Format (Tuple/List) - Backward Compatibility
            elif isinstance(evt, (list, tuple)) and len(evt) >= 2:
                start, end = evt[0], evt[1]
                events.append({
                    'event_type': 'repetition', # Default for legacy
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'confidence': analysis.confidence_score
                })
        
        context = {
            'analysis': analysis,
            'recording': analysis.recording,
            'events': events,
            'total_events': len(events)
        }
        return render(request, 'diagnosis/analysis_detail.html', context)
    except Exception as e:
        messages.error(request, f"Analysis error: {e}")
        return redirect('diagnosis:recordings_list')

@login_required
@require_POST
def upload_recording(request):
    """Handle audio upload with language selection"""
    if 'audio_file' not in request.FILES:
        return JsonResponse({'error': 'No audio file provided'}, status=400)
        
    try:
        patient = request.user.patient_profile
        audio_file = request.FILES['audio_file']
        language = request.POST.get('language', 'english')
        
        # Debug log
        print(f"DEBUG: Uploading '{audio_file.name}' (Language: {language})")
        
        if audio_file.size > settings.MAX_UPLOAD_SIZE:
            return JsonResponse({'error': 'File too large. Max 10MB.'}, status=400)
        
        file_ext = os.path.splitext(audio_file.name)[1].lower()
        if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
            return JsonResponse({'error': 'Invalid format.'}, status=400)
        
        recording = AudioRecording.objects.create(
            patient=patient,
            audio_file=audio_file,
            file_size_bytes=audio_file.size,
            status='pending'
        )
        
        logger.info(f"Audio {recording.id} uploaded by {request.user.username}")
        
        # Pass language to the Celery task
        process_audio_recording.delay(recording.id, language=language)
        
        return JsonResponse({
            'success': True,
            'recording_id': recording.id,
            'message': 'Upload successful. Analyzing...'
        })
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def delete_recording(request, recording_id):
    if request.method != 'POST': return redirect('diagnosis:recordings_list')
    try:
        patient = request.user.patient_profile
        rec = get_object_or_404(AudioRecording, id=recording_id, patient=patient)
        rec.delete()
        messages.success(request, 'Recording deleted')
    except Exception:
        messages.error(request, 'Error deleting recording')
    return redirect('diagnosis:recordings_list')

@login_required
def check_status(request, recording_id):
    try:
        patient = request.user.patient_profile
        rec = get_object_or_404(AudioRecording, id=recording_id, patient=patient)
        data = {'id': rec.id, 'status': rec.status, 'error_message': rec.error_message}
        if rec.status == 'completed' and hasattr(rec, 'analysis'):
            data.update({
                'analysis_id': rec.analysis.id,
                'severity': rec.analysis.severity,
                'mismatch_percentage': rec.analysis.mismatch_percentage,
            })
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=404)