# diagnosis/tasks.py
from celery import shared_task
from django.utils import timezone
from django.conf import settings
import logging
import librosa
import torch
import gc
import os
import tempfile
import subprocess

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
        # Convert uploaded audio to a stable WAV format (16k mono) using ffmpeg if available.
        # This avoids librosa/ffmpeg mismatches for browser blobs (webm/ogg) and ensures
        # consistent sampling rate for the detection model.
        converted_path = None
        use_path = audio_path
        try:
            tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            converted_path = tf.name
            tf.close()

            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ac', '1', '-ar', str(settings.AUDIO_SAMPLE_RATE),
                converted_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            use_path = converted_path
            logger.info(f"Converted audio to WAV for analysis: {converted_path}")
        except Exception as e:
            logger.warning(f"Audio conversion failed or ffmpeg not found, using original file: {e}")

        # Perform the analysis
        # FIX: Changed argument 'audio_file_path' to 'audio_path' to match new class definition
        analysis_data = detector.analyze_audio(
            audio_path=use_path,
            language=language
        )
        
        # 4. Save Results
        # Sanitize analysis_data to ensure JSON serializable types (no numpy/torch types)
        def _sanitize(obj):
            import numpy as _np
            # torch may not be available here; check generically
            try:
                import torch as _torch
            except Exception:
                _torch = None

            if obj is None:
                return None
            # numpy scalar -> python scalar
            if isinstance(obj, (_np.generic,)):
                try:
                    return obj.item()
                except Exception:
                    return obj.tolist() if hasattr(obj, 'tolist') else obj
            # torch scalar/tensor
            if _torch is not None and isinstance(obj, _torch.Tensor):
                try:
                    return _sanitize(obj.detach().cpu().numpy())
                except Exception:
                    return obj.item() if obj.numel() == 1 else _sanitize(obj.tolist())
            # numpy ndarray
            if isinstance(obj, _np.ndarray):
                return _sanitize(obj.tolist())
            # dict
            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            # list/tuple
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            # other python primitives (including bool, int, float, str)
            try:
                # json can't serialize some numpy.bool_ etc, ensure native bool
                if isinstance(obj, bool):
                    return bool(obj)
                if isinstance(obj, (int, float, str)):
                    return obj
            except Exception:
                pass
            return obj

        mismatches_safe = _sanitize(analysis_data.get('mismatched_chars'))
        timestamps_safe = _sanitize(analysis_data.get('stutter_timestamps'))

        # Ensure numeric scalars are native python types
        def _to_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        analysis = AnalysisResult.objects.create(
            recording=recording,
            actual_transcript=str(analysis_data.get('actual_transcript', '')),
            target_transcript=str(analysis_data.get('target_transcript', '')),
            mismatched_chars=mismatches_safe or [],
            mismatch_percentage=_to_float(analysis_data.get('mismatch_percentage', 0.0)),
            ctc_loss_score=_to_float(analysis_data.get('ctc_loss_score', 0.0)),
            stutter_timestamps=timestamps_safe or [],
            total_stutter_duration=_to_float(analysis_data.get('total_stutter_duration', 0.0)),
            stutter_frequency=_to_float(analysis_data.get('stutter_frequency', 0.0)),
            severity=str(analysis_data.get('severity', 'none')),
            confidence_score=_to_float(analysis_data.get('confidence_score', 0.0)),
            analysis_duration_seconds=_to_float(analysis_data.get('analysis_duration_seconds', 0.0)),
            model_version=str(analysis_data.get('model_version', 'unknown'))
        )
        
        # 5. Cleanup & Success
        recording.status = 'completed'
        recording.processed_at = timezone.now()
        recording.save()

        # Remove temporary converted file if one was created
        try:
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
        except Exception:
            pass
        
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