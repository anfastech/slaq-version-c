# diagnosis/ai_engine/detect_stuttering.py
"""
SLAQ AI Engine - Stutter Detection via External ML API
=======================================================

This module provides stuttering analysis using an external ML API endpoint
hosted on HuggingFace. All heavy ML processing is offloaded to the API.

Supports Indian languages through MMS model:
- Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam
- Punjabi, Urdu, Assamese, Odia, Bhojpuri, Maithili, English
"""

# Base API URL for stutter detection
# Points to slaq-version-c-ai-enginee HuggingFace Space
API_URL = "https://anfastech-slaq-version-c-ai-enginee.hf.space"

import logging
import os
import time
import requests
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)


def get_config() -> Dict[str, Any]:
    """Load configuration from Django settings at runtime."""
    try:
        from django.conf import settings
        # Get base URL from settings or use default
        base_url = getattr(settings, 'STUTTER_API_URL', API_URL)
        # Ensure it ends with /analyze for the analyze endpoint
        if not base_url.endswith('/analyze'):
            api_url = base_url.rstrip('/') + '/analyze'
        else:
            api_url = base_url
        
        return {
            'api_url': api_url,
            'api_timeout': getattr(settings, 'STUTTER_API_TIMEOUT', 300),
            'default_language': getattr(settings, 'DEFAULT_LANGUAGE', 'hindi'),
            'sample_rate': getattr(settings, 'AUDIO_SAMPLE_RATE', 16000),
            'max_retries': getattr(settings, 'STUTTER_API_MAX_RETRIES', 3),
            'retry_delay': getattr(settings, 'STUTTER_API_RETRY_DELAY', 5),
        }
    except Exception:
        # Fallback for standalone usage
        return {
            'api_url': API_URL + '/analyze',
            'api_timeout': 300,
            'default_language': 'hindi',
            'sample_rate': 16000,
            'max_retries': 3,
            'retry_delay': 5,
        }


# Indian Language Codes for MMS Model
INDIAN_LANGUAGE_CODES = {
    # Primary Indian Languages
    'hindi': 'hin',
    'tamil': 'tam',
    'telugu': 'tel',
    'bengali': 'ben',
    'marathi': 'mar',
    'gujarati': 'guj',
    'kannada': 'kan',
    'malayalam': 'mal',
    'punjabi': 'pan',
    'urdu': 'urd',
    'assamese': 'asm',
    'odia': 'ory',
    'oriya': 'ory',  # Alias
    
    # Additional Indian Languages
    'bhojpuri': 'bho',
    'maithili': 'mai',
    'sanskrit': 'san',
    'konkani': 'kok',
    'dogri': 'doi',
    'kashmiri': 'kas',
    'sindhi': 'snd',
    'nepali': 'nep',
    'bodo': 'brx',
    'santali': 'sat',
    'manipuri': 'mni',
    
    # English (Indian English supported)
    'english': 'eng',
    
    # Common aliases (3-letter codes map to themselves)
    'hin': 'hin',
    'tam': 'tam',
    'tel': 'tel',
    'ben': 'ben',
    'mar': 'mar',
    'guj': 'guj',
    'kan': 'kan',
    'mal': 'mal',
    'pan': 'pan',
    'urd': 'urd',
    'eng': 'eng',
    'asm': 'asm',
    'ory': 'ory',
}

# List of supported language display names
SUPPORTED_LANGUAGES = [
    'Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi', 'Gujarati',
    'Kannada', 'Malayalam', 'Punjabi', 'Urdu', 'Assamese', 'Odia',
    'Bhojpuri', 'Maithili', 'Sanskrit', 'Konkani', 'Dogri', 'Kashmiri',
    'Sindhi', 'Nepali', 'English'
]


class StutterDetector:
    """
    Stutter detection using external ML API with Indian language support.
    
    API endpoint: Configurable via Django settings or defaults to HuggingFace Space
    
    Attributes:
        api_url: The external API endpoint URL
        api_timeout: Request timeout in seconds
        default_language: Default language for analysis
    """
    
    def __init__(self):
        """Initialize detector - no local models needed, all processing via API."""
        logger.info("🔄 Initializing StutterDetector (API-only mode)")
        
        config = get_config()
        self.api_url = config['api_url']
        self.api_timeout = config['api_timeout']
        self.default_language = config['default_language']
        self.sample_rate = config['sample_rate']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay']
        
        logger.info(f"✅ StutterDetector initialized")
        logger.info(f"   📡 API URL: {self.api_url}")
        logger.info(f"   🌐 Default Language: {self.default_language}")
        logger.info(f"   ⏱️ Timeout: {self.api_timeout}s")
        logger.info(f"   🔄 Max Retries: {self.max_retries}")
    
    def _resolve_language(self, language: Optional[str]) -> str:
        """
        Resolve language name/code to MMS language code.
        
        Args:
            language: Language name or code (e.g., 'hindi', 'hin', 'Hindi')
        
        Returns:
            MMS language code (e.g., 'hin')
        """
        if not language:
            return INDIAN_LANGUAGE_CODES.get(self.default_language, 'hin')
        
        # Normalize to lowercase
        lang_lower = language.lower().strip()
        
        # Handle 'auto' detection
        if lang_lower == 'auto':
            return INDIAN_LANGUAGE_CODES.get(self.default_language, 'hin')
        
        # Direct lookup
        if lang_lower in INDIAN_LANGUAGE_CODES:
            return INDIAN_LANGUAGE_CODES[lang_lower]
        
        # Fuzzy matching for common variations
        for key, code in INDIAN_LANGUAGE_CODES.items():
            if lang_lower.startswith(key[:3]) or key.startswith(lang_lower[:3]):
                return code
        
        # Default fallback
        logger.warning(f"⚠️ Unknown language '{language}', defaulting to Hindi")
        return 'hin'
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported Indian languages."""
        return SUPPORTED_LANGUAGES.copy()
    
    def analyze_audio(
        self,
        audio_path: Optional[str] = None,
        audio_file_path: Optional[str] = None,
        language: Optional[str] = None,
        proper_transcript: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze audio for stuttering using external ML API endpoint.
        
        Supports both parameter naming conventions for backward compatibility:
        - audio_path: New naming convention
        - audio_file_path: Legacy naming convention
        
        Args:
            audio_path: Path to audio file (preferred)
            audio_file_path: Path to audio file (legacy, for backward compatibility)
            language: Language name or code (e.g., 'hindi', 'hin')
            proper_transcript: Optional expected transcript for comparison
            **kwargs: Additional arguments (ignored, for compatibility)
        
        Returns:
            Dictionary with complete analysis results:
            - actual_transcript: Transcribed text from audio
            - target_transcript: Expected transcript (if provided)
            - mismatched_chars: List of character-level mismatches
            - mismatch_percentage: Percentage of mismatched characters
            - ctc_loss_score: CTC loss score from model
            - stutter_timestamps: List of detected stutter events
            - total_stutter_duration: Total duration of stuttering in seconds
            - stutter_frequency: Frequency of stuttering events per minute
            - severity: Severity classification (none, mild, moderate, severe)
            - confidence_score: Overall confidence in the analysis
            - analysis_duration_seconds: Time taken for analysis
            - model_version: Version of the model used
            - language_detected: Detected/used language code
        """
        start_time = time.time()
        
        # Handle both parameter names for backward compatibility
        file_path = audio_path or audio_file_path
        
        if not file_path:
            raise ValueError("Either 'audio_path' or 'audio_file_path' must be provided")
        
        try:
            logger.info(f"🎯 Starting API analysis for: {file_path}")
            
            # Resolve language code
            lang_code = self._resolve_language(language)
            logger.info(f"🌐 Language: {language} -> {lang_code}")
            logger.info(f"📝 Transcript provided: {bool(proper_transcript)}")
            
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"📋 File: {os.path.basename(file_path)}")
            logger.info(f"📋 Size: {file_size:,} bytes")
            logger.info(f"📋 Format: {file_ext}")
            
            # Prepare and send API request with retry logic
            result = None
            last_exception = None
            
            for attempt in range(1, self.max_retries + 1):
                try:
                    with open(file_path, "rb") as f:
                        files = {"audio": (os.path.basename(file_path), f, self._get_mime_type(file_ext))}
                        data = {
                            "transcript": proper_transcript if proper_transcript else "",
                            "language": lang_code,
                        }
                        
                        if attempt > 1:
                            logger.info(f"📤 Retrying API request (attempt {attempt}/{self.max_retries})...")
                        else:
                            logger.info(f"📤 Sending request to API...")
                        logger.debug(f"📤 API URL: {self.api_url}")
                        logger.debug(f"📤 Data: {data}")
                        
                        response = requests.post(
                            self.api_url,
                            files=files,
                            data=data,
                            timeout=self.api_timeout
                        )
                        
                        logger.info(f"📥 Response status: {response.status_code}")
                        response.raise_for_status()
                        
                        result = response.json()
                        logger.info(f"✅ API response received")
                        logger.debug(f"✅ Response keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                        # Log transcript values for debugging
                        if isinstance(result, dict):
                            actual = result.get('actual_transcript', '')
                            target = result.get('target_transcript', '')
                            logger.info(f"📝 Actual transcript length: {len(actual)} chars")
                            logger.info(f"📝 Target transcript length: {len(target)} chars")
                            logger.debug(f"📝 Actual transcript preview: {actual[:100] if actual else '(empty)'}")
                            logger.debug(f"📝 Target transcript preview: {target[:100] if target else '(empty)'}")
                        break  # Success, exit retry loop
                        
                except requests.exceptions.Timeout as e:
                    last_exception = e
                    logger.warning(f"⚠️ API request timed out after {self.api_timeout}s (attempt {attempt}/{self.max_retries})")
                    if attempt < self.max_retries:
                        logger.info(f"⏳ Waiting {self.retry_delay}s before retry...")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"❌ All retry attempts exhausted")
                        raise TimeoutError(f"API request timed out after {self.api_timeout} seconds (tried {self.max_retries} times)")
                        
                except requests.exceptions.ConnectionError as e:
                    last_exception = e
                    logger.warning(f"⚠️ Failed to connect to API: {e} (attempt {attempt}/{self.max_retries})")
                    if attempt < self.max_retries:
                        logger.info(f"⏳ Waiting {self.retry_delay}s before retry...")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"❌ All retry attempts exhausted")
                        raise ConnectionError(f"Failed to connect to analysis API after {self.max_retries} attempts: {e}")
                        
                except requests.exceptions.HTTPError as e:
                    # Don't retry on HTTP errors (4xx, 5xx) unless it's a 503 (service unavailable)
                    status_code = response.status_code if 'response' in locals() else None
                    if status_code == 503 and attempt < self.max_retries:
                        logger.warning(f"⚠️ API returned 503 (Service Unavailable) - retrying (attempt {attempt}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"❌ API returned error: {status_code}")
                        if 'response' in locals() and response.text:
                            logger.error(f"❌ Response: {response.text[:500]}")
                        raise RuntimeError(f"API error ({status_code}): {response.text[:200] if 'response' in locals() and response.text else 'Unknown error'}")
                        
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    logger.warning(f"⚠️ Request failed: {type(e).__name__}: {e} (attempt {attempt}/{self.max_retries})")
                    if attempt < self.max_retries:
                        logger.info(f"⏳ Waiting {self.retry_delay}s before retry...")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"❌ All retry attempts exhausted")
                        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
            
            if result is None:
                raise RuntimeError(f"Failed to get response from API after {self.max_retries} attempts")
            
            # Calculate analysis duration
            analysis_duration = time.time() - start_time
            
            # Format and validate result with defaults
            formatted_result = self._format_result(
                result,
                proper_transcript,
                lang_code,
                analysis_duration
            )
            
            logger.info(f"✅ Analysis complete in {analysis_duration:.2f}s")
            logger.info(f"   📊 Severity: {formatted_result['severity']}")
            logger.info(f"   📊 Confidence: {formatted_result['confidence_score']:.2f}")
            logger.info(f"   📊 Events: {len(formatted_result['stutter_timestamps'])}")
            
            return formatted_result
            
        except (FileNotFoundError, ValueError, TimeoutError, ConnectionError) as e:
            # Re-raise known errors
            raise
        except Exception as e:
            logger.error(f"❌ Analysis failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"Audio analysis failed: {e}") from e
    
    def _get_mime_type(self, extension: str) -> str:
        """Get MIME type for audio file extension."""
        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.ogg': 'audio/ogg',
            '.webm': 'audio/webm',
            '.m4a': 'audio/m4a',
            '.flac': 'audio/flac',
            '.aac': 'audio/aac',
        }
        return mime_types.get(extension.lower(), 'audio/wav')
    
    def _format_result(
        self,
        api_result: Dict[str, Any],
        proper_transcript: str,
        lang_code: str,
        analysis_duration: float
    ) -> Dict[str, Any]:
        """
        Format API result with defaults and validation.
        
        Ensures all required fields are present with proper types.
        """
        # Extract stutter timestamps and ensure proper format
        raw_timestamps = api_result.get('stutter_timestamps', [])
        formatted_timestamps = self._format_timestamps(raw_timestamps)
        
        # Calculate total stutter duration from timestamps if not provided
        total_duration = api_result.get('total_stutter_duration')
        if total_duration is None:
            total_duration = sum(
                (evt.get('end', 0) - evt.get('start', 0))
                for evt in formatted_timestamps
                if isinstance(evt, dict)
            )
        
        # Extract transcripts with proper fallback
        actual_transcript = str(api_result.get('actual_transcript', '')).strip()
        target_transcript = str(api_result.get('target_transcript', '')).strip()
        
        # If target_transcript is empty from API, use proper_transcript if provided
        if not target_transcript and proper_transcript:
            target_transcript = proper_transcript.strip()
        
        # Log for debugging
        logger.info(f"📝 Final transcripts - Actual: {len(actual_transcript)} chars, Target: {len(target_transcript)} chars")
        
        return {
            # Transcription
            'actual_transcript': actual_transcript,
            'target_transcript': target_transcript,
            
            # Mismatch analysis
            'mismatched_chars': api_result.get('mismatched_chars', []),
            'mismatch_percentage': self._safe_float(api_result.get('mismatch_percentage', 0.0)),
            
            # Model scores
            'ctc_loss_score': self._safe_float(api_result.get('ctc_loss_score', 0.0)),
            
            # Stutter events
            'stutter_timestamps': formatted_timestamps,
            'total_stutter_duration': self._safe_float(total_duration, 0.0),
            'stutter_frequency': self._safe_float(api_result.get('stutter_frequency', 0.0)),
            
            # Classification
            'severity': str(api_result.get('severity', 'none')).lower(),
            'confidence_score': self._safe_float(api_result.get('confidence_score', 0.0)),
            
            # Metadata
            'analysis_duration_seconds': round(analysis_duration, 2),
            'model_version': str(api_result.get('model_version', 'external-api-v1')),
            'language_detected': lang_code,
        }
    
    def _format_timestamps(self, raw_timestamps: Any) -> List[Dict[str, Any]]:
        """
        Format stutter timestamps to consistent structure.
        
        Handles various input formats:
        - List of dicts: [{'start': 0.5, 'end': 1.0, 'type': 'repetition'}, ...]
        - List of tuples: [(0.5, 1.0), (2.0, 2.5), ...]
        - List of lists: [[0.5, 1.0], [2.0, 2.5], ...]
        """
        if not raw_timestamps:
            return []
        
        formatted = []
        for i, evt in enumerate(raw_timestamps):
            if isinstance(evt, dict):
                # Already in dict format
                formatted.append({
                    'type': str(evt.get('type', evt.get('event_type', 'dysfluency'))),
                    'start': self._safe_float(evt.get('start', evt.get('start_time', 0))),
                    'end': self._safe_float(evt.get('end', evt.get('end_time', 0))),
                    'duration': self._safe_float(
                        evt.get('duration', 
                               evt.get('end', 0) - evt.get('start', 0))
                    ),
                    'confidence': self._safe_float(evt.get('confidence', evt.get('probability', 0.5))),
                    'text': str(evt.get('text', '')),
                })
            elif isinstance(evt, (list, tuple)) and len(evt) >= 2:
                # Tuple/list format: (start, end) or (start, end, type)
                start = self._safe_float(evt[0])
                end = self._safe_float(evt[1])
                evt_type = str(evt[2]) if len(evt) > 2 else 'dysfluency'
                formatted.append({
                    'type': evt_type,
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'confidence': 0.5,
                    'text': '',
                })
            else:
                logger.warning(f"⚠️ Skipping invalid timestamp format at index {i}: {evt}")
        
        return formatted
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def check_api_health(self) -> Dict[str, Any]:
        """
        Check if the API endpoint is healthy and accessible.
        
        Returns:
            Dictionary with health status information:
            - healthy: bool - Whether the API is accessible
            - status_code: int - HTTP status code
            - message: str - Status message
            - response_time: float - Response time in seconds
        """
        health_url = self.api_url.replace('/analyze', '/health')
        if health_url == self.api_url:  # If no /health endpoint, try root
            health_url = self.api_url.rsplit('/analyze', 1)[0] + '/health'
        
        try:
            start_time = time.time()
            response = requests.get(health_url, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return {
                    'healthy': True,
                    'status_code': response.status_code,
                    'message': 'API is healthy and accessible',
                    'response_time': round(response_time, 2),
                    'details': response.json() if response.text else {}
                }
            else:
                return {
                    'healthy': False,
                    'status_code': response.status_code,
                    'message': f'API returned status {response.status_code}',
                    'response_time': round(response_time, 2)
                }
        except requests.exceptions.Timeout:
            return {
                'healthy': False,
                'status_code': None,
                'message': 'Health check timed out',
                'response_time': None
            }
        except requests.exceptions.ConnectionError as e:
            return {
                'healthy': False,
                'status_code': None,
                'message': f'Failed to connect to API: {e}',
                'response_time': None
            }
        except Exception as e:
            return {
                'healthy': False,
                'status_code': None,
                'message': f'Health check failed: {e}',
                'response_time': None
            }


# Alias for backward compatibility with model_loader.py
AdvancedStutterDetector = StutterDetector
