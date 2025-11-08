# reports/models.py
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from diagnosis.models import AnalysisResult


class Report(models.Model):
    """Generated report for doctor review"""
    
    REPORT_TYPES = [
        ('session', 'Session Report'),
        ('weekly', 'Weekly Progress'),
        ('monthly', 'Monthly Summary'),
    ]
    
    patient = models.ForeignKey('core.Patient', on_delete=models.CASCADE, related_name='reports')
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES)
    analyses = models.ManyToManyField(AnalysisResult, related_name='reports')
    
    # Report Content
    summary = models.TextField()
    key_findings = models.JSONField(default=dict)
    progress_metrics = models.JSONField(default=dict)
    recommendations = models.TextField()
    
    # Metadata
    generated_at = models.DateTimeField(auto_now_add=True)
    pdf_file = models.FileField(upload_to='reports/%Y/%m/', blank=True, null=True)
    shared_with_therapist = models.BooleanField(default=False)
    therapist_notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-generated_at']
        indexes = [
            models.Index(fields=['patient', '-generated_at']),
        ]
    
    def __str__(self):
        return f"{self.report_type} Report - {self.patient.user.username} - {self.generated_at.strftime('%Y-%m-%d')}"


class TherapyRecommendation(models.Model):
    """Personalized therapy exercises and recommendations"""
    
    DIFFICULTY_LEVELS = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]
    
    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name='therapy_recommendations')
    exercise_title = models.CharField(max_length=200)
    description = models.TextField()
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_LEVELS)
    duration_minutes = models.IntegerField()
    frequency_per_week = models.IntegerField()
    instructions = models.TextField()
    video_url = models.URLField(blank=True)
    
    # Tracking
    assigned_at = models.DateTimeField(auto_now_add=True)
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['difficulty', '-assigned_at']
    
    def __str__(self):
        return f"{self.exercise_title} ({self.difficulty})"


class ProgressTracking(models.Model):
    """Track patient progress over time"""
    
    patient = models.ForeignKey('core.Patient', on_delete=models.CASCADE, related_name='progress_history')
    recorded_date = models.DateField()
    
    # Aggregated Metrics
    avg_mismatch_percentage = models.FloatField()
    avg_ctc_loss = models.FloatField()
    avg_stutter_frequency = models.FloatField()
    total_practice_minutes = models.IntegerField(default=0)
    
    # Improvement Indicators
    improvement_score = models.FloatField(
        validators=[MinValueValidator(-100.0), MaxValueValidator(100.0)],
        help_text="Positive = improvement, Negative = decline"
    )
    
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-recorded_date']
        unique_together = ['patient', 'recorded_date']
        indexes = [
            models.Index(fields=['patient', '-recorded_date']),
        ]
    
    def __str__(self):
        return f"Progress {self.patient.user.username} - {self.recorded_date}"
