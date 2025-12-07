# Django App - AI Engine Integration

## Architecture: API Client Pattern

This Django app acts as a **thin client** that calls the AI Engine API service. No ML models are loaded locally - all processing happens via HTTP requests to the AI Engine service.

---

## Architecture Diagram

```
┌─────────────────────────────────┐
│   Django App (slaq-version-c)   │
│                                 │
│  ┌───────────────────────────┐  │
│  │  model_loader.py          │  │
│  │  (API Client Singleton)   │  │
│  └───────────────────────────┘  │
│           │                      │
│  ┌───────────────────────────┐  │
│  │  detect_stuttering.py     │  │
│  │  (StutterDetector API)    │  │
│  └───────────────────────────┘  │
│           │                      │
│  ┌───────────────────────────┐  │
│  │  features.py              │  │
│  │  (ASRFeatureExtractor)    │  │
│  └───────────────────────────┘  │
└───────────┼──────────────────────┘
            │ HTTP Requests
            │ (REST API)
            ▼
┌─────────────────────────────────┐
│  AI Engine Service              │
│  (slaq-version-c-ai-enginee)    │
│                                 │
│  ┌───────────────────────────┐  │
│  │  IndicWav2Vec Hindi       │  │
│  │  (ML Models)              │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

---

## Module Responsibilities

### 1. `model_loader.py`
**Purpose**: Singleton pattern for API client management

- Provides `get_stutter_detector()` function
- Returns API client instance (not a local model)
- Handles singleton pattern for efficient reuse
- No local models loaded

**Usage:**
```python
from diagnosis.ai_engine.model_loader import get_stutter_detector

detector = get_stutter_detector()  # Returns API client
result = detector.analyze_audio(audio_path, transcript)
```

---

### 2. `detect_stuttering.py`
**Purpose**: API client for stutter detection

- `StutterDetector` class (API client)
- Makes HTTP requests to AI Engine service
- Handles retries, timeouts, error handling
- Formats API responses for Django app

**Key Methods:**
- `analyze_audio()` - Main analysis via API
- `check_api_health()` - Health check endpoint

**API Endpoint**: `{STUTTER_API_URL}/analyze`

---

### 3. `features.py`
**Purpose**: Feature extraction via API

- `ASRFeatureExtractor` class (API client)
- Calls AI Engine API for feature extraction
- No local BERT or Wav2Vec2 models
- Returns features from API response

**Key Methods:**
- `get_transcription_features()` - Get ASR features via API
- `get_audio_features()` - Get audio metadata + features

**Backward Compatibility:**
- `HybridFeatureExtractor` alias maintained for legacy code

---

## Configuration

### Django Settings

```python
# settings.py

# AI Engine API Configuration
STUTTER_API_URL = "https://anfastech-slaq-version-c-ai-enginee.hf.space"
STUTTER_API_TIMEOUT = 300  # seconds
STUTTER_API_MAX_RETRIES = 3
STUTTER_API_RETRY_DELAY = 5  # seconds
DEFAULT_LANGUAGE = 'hindi'
AUDIO_SAMPLE_RATE = 16000
```

---

## Data Flow

### 1. Audio Upload
```
User uploads audio
    ↓
Django saves file
    ↓
Celery task triggered
```

### 2. Analysis Request
```
Celery task
    ↓
get_stutter_detector() [model_loader.py]
    ↓
StutterDetector.analyze_audio() [detect_stuttering.py]
    ↓
HTTP POST to AI Engine API
    ↓
AI Engine processes with IndicWav2Vec Hindi
    ↓
JSON response returned
    ↓
Django saves results to database
```

### 3. Feature Extraction (if needed)
```
Code needs features
    ↓
ASRFeatureExtractor.get_transcription_features() [features.py]
    ↓
HTTP POST to AI Engine API
    ↓
AI Engine extracts features
    ↓
Features returned in response
```

---

## Benefits

### ✅ Separation of Concerns
- Django app: Web application logic
- AI Engine: ML/AI processing
- Clear boundaries between services

### ✅ Scalability
- AI Engine can scale independently
- Django app doesn't need GPU resources
- Can deploy AI Engine on specialized hardware

### ✅ Maintainability
- ML model updates don't require Django deployment
- Easier to test and debug
- Clear API contract

### ✅ Resource Efficiency
- No ML models loaded in Django app
- Reduced memory footprint
- Faster Django app startup

---

## Error Handling

All API clients include:
- Retry logic with exponential backoff
- Timeout handling
- Connection error handling
- Detailed logging

---

## Testing

### Mock API Responses
```python
from unittest.mock import patch, Mock

@patch('diagnosis.ai_engine.detect_stuttering.requests.post')
def test_analyze_audio(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        'actual_transcript': 'test',
        'severity': 'none'
    }
    mock_post.return_value = mock_response
    
    detector = get_stutter_detector()
    result = detector.analyze_audio('test.wav')
    assert result['actual_transcript'] == 'test'
```

---

## Migration Notes

### Before (Local Models)
- Models loaded in Django app
- High memory usage
- Slow startup time
- Tight coupling

### After (API Client)
- ✅ No local models
- ✅ Low memory usage
- ✅ Fast startup
- ✅ Loose coupling via API

---

## API Contract

### Request Format
```http
POST /analyze
Content-Type: multipart/form-data

audio: <file>
transcript: <string>
language: <string>
```

### Response Format
```json
{
  "actual_transcript": "transcribed text",
  "target_transcript": "expected text",
  "severity": "none|mild|moderate|severe",
  "confidence_score": 0.85,
  "stutter_timestamps": [...],
  "model_version": "indicwav2vec-hindi-asr-v1"
}
```

---

This architecture ensures the Django app remains lightweight while leveraging the full power of the AI Engine service.

