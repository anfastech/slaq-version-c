# diagnosis/tasks.py
from celery import shared_task
from django.utils import timezone
from django.conf import settings
import logging
import librosa
import torch
import gc
import os

from .models import AudioRecording, AnalysisResult
from .ai_engine.model_loader import get_stutter_detector

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def process_audio_recording(self, recording_id, language='english'):
    """
    Async task to process audio recording using Meta MMS-1B.
    """
    try:
        logger.info(f"🎯 Processing recording {recording_id} [Language: {language}]")
        
        # 1. Retrieve Recording
        try:
            recording = AudioRecording.objects.get(id=recording_id)
        except AudioRecording.DoesNotExist:
            logger.error(f"❌ Recording {recording_id} not found")
            return None

        # Update status to processing
        recording.status = 'processing'
        recording.save()
        
        # 2. Pre-analysis Checks
        audio_path = recording.audio_file.path
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        # Calculate duration if missing
        try:
            duration = librosa.get_duration(path=audio_path)
            recording.duration_seconds = round(duration, 2)
            recording.save()
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate duration: {e}")

        # 3. Run AI Analysis (MMS-1B)
        logger.info(f"🤖 Invoking MMS-1B Stutter Detector...")
        detector = get_stutter_detector()
        
        # Perform the analysis
        # FIX: Changed argument 'audio_file_path' to 'audio_path' to match new class definition
        analysis_data = detector.analyze_audio(
            audio_path=audio_path,
            language=language
        )
        
        # 4. Save Results
        analysis = AnalysisResult.objects.create(
            recording=recording,
            actual_transcript=analysis_data['actual_transcript'],
            target_transcript=analysis_data['target_transcript'],
            mismatched_chars=analysis_data['mismatched_chars'],
            mismatch_percentage=analysis_data['mismatch_percentage'],
            ctc_loss_score=analysis_data['ctc_loss_score'],
            stutter_timestamps=analysis_data['stutter_timestamps'],
            total_stutter_duration=analysis_data['total_stutter_duration'],
            stutter_frequency=analysis_data['stutter_frequency'],
            severity=analysis_data['severity'],
            confidence_score=analysis_data['confidence_score'],
            analysis_duration_seconds=analysis_data['analysis_duration_seconds'],
            model_version=analysis_data['model_version']
        )
        
        # 5. Cleanup & Success
        recording.status = 'completed'
        recording.processed_at = timezone.now()
        recording.save()
        
        logger.info(f"✅ Recording {recording_id} processed successfully")
        
        return {
            'recording_id': recording_id,
            'status': 'completed',
            'language': language
        }
        
    except Exception as e:
        logger.error(f"❌ Processing failed for recording {recording_id}: {e}")
        
        # Update DB status
        try:
            recording = AudioRecording.objects.get(id=recording_id)
            recording.status = 'failed'
            recording.error_message = str(e)
            recording.save()
        except:
            pass
            
        # GPU Memory Cleanup on Failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # Retry logic for transient errors
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))
        
    finally:
        # Always try to clear cache after a heavy 1B parameter run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()