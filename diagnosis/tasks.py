# diagnosis/tasks.py
from celery import shared_task
from django.utils import timezone
from django.conf import settings
import logging
# import librosa  # TODO: Install ML libraries (torch, librosa, transformers)

from .models import AudioRecording, AnalysisResult, StutterEvent
# from .ai_engine.model_loader import get_stutter_detector  # TODO: Re-enable when ML libs installed

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_audio_recording(self, recording_id):
    """
    Async task to process audio recording
    Following SLAQ AI Workflow:
    1. Audio Input
    2. AI Diagnosis (Articulation analysis)
    3. Report Generation (Results storage)
    """
    
    try:
        logger.info(f"🎯 Processing recording {recording_id}")
        
        # Get recording
        recording = AudioRecording.objects.get(id=recording_id)
        recording.status = 'processing'
        recording.save()
        
        # Get audio file path
        audio_path = recording.audio_file.path
        
        # Calculate duration
        try:
            duration = librosa.get_duration(path=audio_path)
            recording.duration_seconds = round(duration, 2)
            recording.save()
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate duration: {e}")
        
        # Load AI detector (singleton - models already loaded)
        detector = get_stutter_detector()
        
        # Run analysis
        analysis_data = detector.analyze_audio(audio_path)
        
        # Save analysis results
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
        
        # Create individual stutter events
        create_stutter_events(analysis, analysis_data)
        
        # Update recording status
        recording.status = 'completed'
        recording.processed_at = timezone.now()
        recording.save()
        
        logger.info(f"✅ Recording {recording_id} processed successfully")
        
        # Trigger report generation if needed
        generate_session_report.delay(recording.patient.id, analysis.id)
        
        return {
            'recording_id': recording_id,
            'analysis_id': analysis.id,
            'severity': analysis.severity,
            'mismatch_percentage': analysis.mismatch_percentage
        }
        
    except AudioRecording.DoesNotExist:
        logger.error(f"❌ Recording {recording_id} not found")
        raise
    
    except Exception as e:
        logger.error(f"❌ Processing failed for recording {recording_id}: {e}")
        
        try:
            recording = AudioRecording.objects.get(id=recording_id)
            recording.status = 'failed'
            recording.error_message = str(e)
            recording.save()
        except:
            pass
        
        # Retry task
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))


def create_stutter_events(analysis, analysis_data):
    """Create individual stutter events from timestamps"""
    try:
        events = []
        
        for start_time, end_time in analysis_data['stutter_timestamps']:
            duration = end_time - start_time
            
            # Determine event type based on duration
            if duration > 0.8:
                event_type = 'prolongation'
            elif duration > 0.4:
                event_type = 'block'
            else:
                event_type = 'repetition'
            
            # Extract affected text from transcript
            # This is simplified - could be enhanced with more precise mapping
            affected_text = analysis.actual_transcript[:20] if analysis.actual_transcript else ""
            
            event = StutterEvent(
                analysis=analysis,
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                affected_text=affected_text,
                confidence=0.85  # Could be calculated more precisely
            )
            events.append(event)
        
        # Bulk create for efficiency
        if events:
            StutterEvent.objects.bulk_create(events)
            logger.info(f"✅ Created {len(events)} stutter events")
    
    except Exception as e:
        logger.error(f"❌ Failed to create stutter events: {e}")


@shared_task
def generate_session_report(patient_id, analysis_id):
    """
    Generate session report after analysis
    Part of SLAQ workflow: Report Generation step
    """
    from reports.models import Report, TherapyRecommendation
    from core.models import Patient
    
    try:
        logger.info(f"📄 Generating session report for patient {patient_id}")
        
        patient = Patient.objects.get(id=patient_id)
        analysis = AnalysisResult.objects.get(id=analysis_id)
        
        # Generate summary
        summary = generate_summary(analysis)
        
        # Calculate key findings
        key_findings = {
            'primary_issue': determine_primary_issue(analysis),
            'improvement_areas': identify_improvement_areas(analysis),
            'strengths': identify_strengths(analysis)
        }
        
        # Progress metrics
        progress_metrics = calculate_progress_metrics(patient)
        
        # Generate recommendations
        recommendations_text = generate_recommendations(analysis, patient)
        
        # Create report
        report = Report.objects.create(
            patient=patient,
            report_type='session',
            summary=summary,
            key_findings=key_findings,
            progress_metrics=progress_metrics,
            recommendations=recommendations_text
        )
        
        report.analyses.add(analysis)
        
        # Create therapy recommendations
        create_therapy_recommendations(report, analysis)
        
        logger.info(f"✅ Session report {report.id} generated")
        
        return report.id
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        raise


def generate_summary(analysis):
    """Generate human-readable summary"""
    severity = analysis.get_severity_display()
    mismatch = analysis.mismatch_percentage
    
    if analysis.severity == 'none':
        return f"No significant stuttering detected. Speech clarity: {100-mismatch:.1f}%"
    
    return (
        f"{severity} stuttering detected with {mismatch:.1f}% speech disfluency. "
        f"Analysis identified {len(analysis.stutter_timestamps)} stutter events "
        f"totaling {analysis.total_stutter_duration:.2f} seconds. "
        f"Frequency: {analysis.stutter_frequency:.1f} stutters per minute."
    )


def determine_primary_issue(analysis):
    """Identify primary stuttering pattern"""
    events = analysis.events.all()
    
    if not events.exists():
        return "No significant issues detected"
    
    # Count event types
    event_counts = {}
    for event in events:
        event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
    
    # Find most common
    primary = max(event_counts.items(), key=lambda x: x[1])
    return f"Primary issue: {primary[0].title()} ({primary[1]} occurrences)"


def identify_improvement_areas(analysis):
    """Suggest areas for improvement"""
    areas = []
    
    if analysis.mismatch_percentage > 30:
        areas.append("Speech fluency and clarity")
    
    if analysis.stutter_frequency > 5:
        areas.append("Reducing frequency of disfluencies")
    
    if analysis.total_stutter_duration > 10:
        areas.append("Shortening duration of stutter events")
    
    return areas if areas else ["Continue current progress"]


def identify_strengths(analysis):
    """Identify positive aspects"""
    strengths = []
    
    if analysis.mismatch_percentage < 15:
        strengths.append("High speech clarity")
    
    if analysis.stutter_frequency < 3:
        strengths.append("Low stutter frequency")
    
    if analysis.confidence_score > 0.8:
        strengths.append("Consistent speech patterns")
    
    return strengths if strengths else ["Working towards improvement"]


def calculate_progress_metrics(patient):
    """Calculate progress over time"""
    from django.db.models import Avg
    from datetime import timedelta
    from django.utils import timezone
    
    now = timezone.now()
    
    # Get all analyses
    all_analyses = AnalysisResult.objects.filter(
        recording__patient=patient
    ).order_by('created_at')
    
    if all_analyses.count() < 2:
        return {'status': 'insufficient_data'}
    
    # Compare first vs latest
    first = all_analyses.first()
    latest = all_analyses.last()
    
    improvement = {
        'mismatch_change': first.mismatch_percentage - latest.mismatch_percentage,
        'frequency_change': first.stutter_frequency - latest.stutter_frequency,
        'total_sessions': all_analyses.count(),
        'trend': 'improving' if latest.mismatch_percentage < first.mismatch_percentage else 'needs_focus'
    }
    
    return improvement


def generate_recommendations(analysis, patient):
    """Generate personalized recommendations"""
    recommendations = []
    
    severity = analysis.severity
    
    if severity in ['moderate', 'severe']:
        recommendations.append("Continue daily practice sessions (15-20 minutes)")
        recommendations.append("Focus on slow, controlled breathing before speaking")
    
    if analysis.stutter_frequency > 5:
        recommendations.append("Practice phrase repetition exercises")
    
    if analysis.total_stutter_duration > 10:
        recommendations.append("Work on prolongation reduction techniques")
    
    recommendations.append("Record progress weekly to track improvement")
    
    return "\n".join(f"• {rec}" for rec in recommendations)


def create_therapy_recommendations(report, analysis):
    """Create specific therapy exercises"""
    from reports.models import TherapyRecommendation
    
    exercises = []
    
    # Breathing exercises (always recommend)
    exercises.append({
        'title': 'Diaphragmatic Breathing',
        'description': 'Practice controlled breathing to improve speech fluency',
        'difficulty': 'beginner',
        'duration': 10,
        'frequency': 7,
        'instructions': (
            "1. Sit comfortably with back straight\n"
            "2. Place one hand on chest, one on stomach\n"
            "3. Breathe deeply through nose (stomach should rise)\n"
            "4. Exhale slowly through mouth\n"
            "5. Repeat for 10 minutes daily"
        )
    })
    
    # Severity-specific exercises
    if analysis.severity in ['moderate', 'severe']:
        exercises.append({
            'title': 'Slow Speech Practice',
            'description': 'Practice speaking at a reduced rate',
            'difficulty': 'intermediate',
            'duration': 15,
            'frequency': 5,
            'instructions': (
                "1. Choose a short paragraph\n"
                "2. Read it at half your normal speed\n"
                "3. Pause between words\n"
                "4. Focus on smooth transitions\n"
                "5. Gradually increase speed while maintaining fluency"
            )
        })
    
    # Create recommendations
    for ex in exercises:
        TherapyRecommendation.objects.create(
            report=report,
            **ex
        )
    
    logger.info(f"✅ Created {len(exercises)} therapy recommendations")


# slaq_project/celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'slaq_project.settings')

app = Celery('slaq_project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')