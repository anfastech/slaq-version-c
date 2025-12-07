"""
Feature extraction via AI Engine API

This module provides feature extraction by calling the external AI Engine API.
All ML model processing is handled by the AI Engine service, not locally.

Architecture: Django App (Client) → AI Engine API (Service)
"""
import logging
import os
import requests
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _get_api_config() -> Dict[str, Any]:
    """Get API configuration from Django settings."""
    try:
        from django.conf import settings
        base_url = getattr(settings, 'STUTTER_API_URL', 'https://anfastech-slaq-version-c-ai-enginee.hf.space')
        return {
            'api_url': base_url.rstrip('/'),
            'api_timeout': getattr(settings, 'STUTTER_API_TIMEOUT', 300),
            'sample_rate': getattr(settings, 'AUDIO_SAMPLE_RATE', 16000),
        }
    except Exception:
        return {
            'api_url': 'https://anfastech-slaq-version-c-ai-enginee.hf.space',
            'api_timeout': 300,
            'sample_rate': 16000,
        }


class ASRFeatureExtractor:
    """
    Feature extractor that calls the AI Engine API for ASR features.
    
    This is a client wrapper that delegates all ML processing to the AI Engine service.
    No local models are loaded - all processing happens via API calls.
    
    Model: ai4bharat/indicwav2vec-hindi (handled by AI Engine service)
    """
    
    def __init__(self):
        """Initialize API client - no local models loaded."""
        self.config = _get_api_config()
        self.api_url = self.config['api_url']
        self.api_timeout = self.config['api_timeout']
        self.sample_rate = self.config['sample_rate']
        logger.info(f"✅ ASRFeatureExtractor initialized (API client mode)")
        logger.info(f"   📡 API URL: {self.api_url}")
    
    def get_transcription_features(
        self, 
        audio_path: str, 
        transcript: Optional[str] = None,
        language: str = 'hindi'
    ) -> Dict[str, Any]:
        """
        Get transcription features from AI Engine API.
        
        Args:
            audio_path: Path to audio file
            transcript: Optional expected transcript
            language: Language code (default: 'hindi')
            
        Returns:
            Dictionary containing:
            - transcript: Transcribed text
            - confidence: Average confidence score
            - word_timestamps: Word-level timestamps
            - frame_confidence: Per-frame confidence (if available)
        """
        try:
            # Call AI Engine API for transcription
            with open(audio_path, "rb") as f:
                files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
                data = {
                    "transcript": transcript if transcript else "",
                    "language": language,
                }
                
                response = requests.post(
                    f"{self.api_url}/analyze",
                    files=files,
                    data=data,
                    timeout=self.api_timeout
                )
                response.raise_for_status()
                result = response.json()
            
            # Extract features from API response
            return {
                'transcript': result.get('actual_transcript', ''),
                'confidence': result.get('confidence_score', 0.0),
                'word_timestamps': result.get('stutter_timestamps', []),
                'ctc_loss': result.get('ctc_loss_score', 0.0),
                'model_version': result.get('model_version', 'unknown')
            }
        except Exception as e:
            logger.error(f"❌ Error getting transcription features from API: {e}")
            raise
    
    def get_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Get audio features from AI Engine API.
        
        This is a simplified version that returns basic audio metadata.
        For full feature extraction, use get_transcription_features().
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata and API response
        """
        try:
            # Get transcription features (includes audio processing)
            features = self.get_transcription_features(audio_path)
            
            # Add audio metadata
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr
            
            return {
                **features,
                'duration': duration,
                'sample_rate': sr,
                'num_samples': len(audio)
            }
        except Exception as e:
            logger.error(f"❌ Error getting audio features: {e}")
            raise


# Alias for backward compatibility
HybridFeatureExtractor = ASRFeatureExtractor