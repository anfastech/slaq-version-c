# diagnosis/models.py
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
from core.models import Patient
import os

def audio_upload_path(instance, filename):
    """Generate upload path: recordings/patient_id/year/month/filename"""
    now = timezone.now()
    # Ensure we handle cases where patient might not be set (though it should be)
    patient_id = instance.patient.id if instance.patient else 'unknown'
    return f'recordings/{patient_id}/{now.year}/{now.month}/{filename}'

class AudioRecording(models.Model):
    """Audio recording submitted by patient"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending Analysis'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='recordings')
    audio_file = models.FileField(upload_to=audio_upload_path)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # File Metadata
    duration_seconds = models.FloatField(null=True, blank=True)
    file_size_bytes = models.IntegerField(null=True, blank=True)
    
    # Timestamps
    recorded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Error Tracking
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-recorded_at']
        indexes = [
            models.Index(fields=['patient', '-recorded_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"Recording {self.id} - {self.recorded_at.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def filename(self):
        return os.path.basename(self.audio_file.name)
    
    def delete(self, *args, **kwargs):
        """Delete audio file when model is deleted"""
        if self.audio_file:
            if os.path.isfile(self.audio_file.path):
                try:
                    os.remove(self.audio_file.path)
                except OSError:
                    pass
        super().delete(*args, **kwargs)


class AnalysisResult(models.Model):
    """AI analysis results from stutter detection (MMS-1B Optimized)"""
    
    SEVERITY_CHOICES = [
        ('none', 'No Stuttering'),
        ('mild', 'Mild'),
        ('moderate', 'Moderate'),
        ('severe', 'Severe'),
    ]
    
    recording = models.OneToOneField(AudioRecording, on_delete=models.CASCADE, related_name='analysis')
    
    # Transcription Results
    actual_transcript = models.TextField(help_text="What the AI heard")
    target_transcript = models.TextField(help_text="What should have been said")
    
    # Stutter Detection Metrics
    mismatched_chars = models.JSONField(
        default=list,
        help_text="List of stuttered character sequences"
    )
    mismatch_percentage = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Percentage of mismatched characters"
    )
    ctc_loss_score = models.FloatField(
        help_text="CTC Loss score - lower is better"
    )
    
    # Advanced Timing Metrics (Restored for MMS System)
    stutter_timestamps = models.JSONField(
        default=list,
        help_text="List of (start, end) timestamp tuples for stutter events"
    )
    total_stutter_duration = models.FloatField(
        default=0.0,
        help_text="Total duration of stuttering in seconds"
    )
    stutter_frequency = models.FloatField(
        default=0.0,
        help_text="Stutters per minute"
    )

    # Overall Assessment
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0
    )
    
    # Metadata
    analysis_duration_seconds = models.FloatField(help_text="How long analysis took")
    model_version = models.CharField(max_length=100, default="facebook/mms-1b-all")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recording']),
            models.Index(fields=['severity']),
        ]
    
    def __str__(self):
        return f"Analysis {self.id} - {self.severity} - {self.mismatch_percentage:.1f}%"
    
    @property
    def is_stuttering_detected(self):
        return self.severity != 'none'