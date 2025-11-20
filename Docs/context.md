
```mermaid

graph TD
    subgraph SLAQ MVP - Django Monolithic Application
        PL[Presentation Layer]
        VL[Views Layer]
        BL[Business Logic]
        
        subgraph PL [Presentation Layer]
            PL1[HTML Templates<br/>Django Template Language]

            <!--
              Updated project context & architecture document
              - Focus: clear architecture, AI model flow, key files, dev + deployment notes
              - Maintainer: SLAQ Research team
              - Last updated: 2025-11-20
            -->

            ```mermaid
            graph TD
                Web[User Browser]
                Web -->|Upload / Record| Frontend[Frontend (Templates + JS)]
                Frontend --> Backend[Django App (views, forms, APIs)]
                Backend --> DB[(PostgreSQL)]
                Backend -->|enqueue task| Broker[Redis]
                Broker -->|push| Worker[Celery Worker(s)]
                Worker --> AI[AI Engine (Wav2Vec2 & pipeline)]
                AI --> Worker
                Worker --> Backend
                Backend --> Frontend
                Backend --> Media[Media Storage (local or S3)]
            ```

            ══════════════════════════════════════════════════════════════════════
                                    SLAQ — Project Context & Overview
            ══════════════════════════════════════════════════════════════════════

            Purpose
            - SLAQ is a Django-based system for detecting stuttering events in recorded speech, computing severity metrics, and tracking progress over time for therapy.

            Audience
            - Developers, ML engineers, clinicians reviewing model outputs, and ops engineers deploying the system.

            Scope of this document
            - Describe architecture, AI model pipeline, how models are loaded and invoked, key files to inspect, dev setup, endpoints, and testing guidance.

            Core architecture
            - Monolithic Django backend serving pages, APIs and enqueueing background analysis tasks.
            - Celery workers perform long-running AI tasks (transcription, event detection, post-processing).
            - Redis is used as the broker/backing queue; PostgreSQL stores user, recording and analysis metadata. Media files stored on disk or S3.

            Key components and their roles
            - `Frontend (templates + JS)` — recording UI, progress indicators, polling endpoints for task status, Chart.js visualizations.
            - `Django Backend` — request handling, user/session management, file upload/validation (`core/`, `diagnosis/`, `reports/`).
            - `Celery Workers` — consume tasks to run AI analysis. Use `--pool=solo` on Windows development, regular pools on Linux.
            - `AI Engine` — model inference and analysis pipeline implemented under `diagnosis/ai_engine/`.

            AI model pipeline (high level)
            1. Upload/Record: user produces an audio file (WAV/MP3/WebM). Backend saves it and creates an `AudioRecording` with status `pending`.
            2. Enqueue task: backend enqueues a Celery task `diagnosis.tasks.analyze_recording({recording_id})`.
            3. Worker: Celery worker picks up the task, moves status to `processing` and loads the AI pipeline.
            4. Preprocessing: audio is validated, resampled (16kHz mono), normalized, and stored temporarily if needed.
            5. ASR (transcription): Wav2Vec2 model (via HuggingFace Transformers or a wrapped local loader) produces logits → greedy/beam decode → transcript.
            6. Alignment & CTC analysis: compute CTC loss and align model output with expected target transcript (if provided) to find mismatches.
            7. Stutter event detection: using heuristics and/or a small classifier, mark timestamps for repetitions, prolongations, blocks and interjections. Implemented in `detect_stuttering.py`.
            8. Postprocessing: compute metrics (mismatch %, stutter frequency, durations, severity score, confidence), format results and persist as `AnalysisResult` and `StutterEvent` records.
            9. Notification: update DB record and optionally trigger webhooks or notifications. Frontend polls `GET /api/analyses/<id>/status/` and `GET /api/analyses/<id>/data/` to display results.

            Model-loading and key files
            - `diagnosis/ai_engine/model_loader.py` — central loader for ML models. Responsibilities:
              - Load model weights (HF Transformers or torchscript)
              - Ensure model is placed on CPU/GPU based on env config
              - Provide a small inference wrapper with batching and streaming support
            - `diagnosis/ai_engine/detect_stuttering.py` — logic to identify stutter events given transcript, logits and timing information.
            - `diagnosis/ai_engine/utils.py` — audio preprocessing helpers, sample rate conversion, normalization, and feature extraction helpers.
            - `diagnosis/tasks.py` — Celery tasks such as `analyze_recording` that orchestrate preprocessing → inference → postprocessing → persistence.

            Model considerations
            - Default ASR: `facebook/wav2vec2-base-960h` for initial transcription.
            - Fine-tuning: recommended approach — create a dataset of labeled utterances with stutter annotations and fine-tune Wav2Vec2 for improved timestamps and token confidence.
            - Hardware: CPU works for dev testing; GPU (CUDA) recommended for reasonable throughput in production or batch processing.

            Data flow (simplified)
            - Browser → POST `/recordings/upload/` → Django saves to `MEDIA_ROOT/recordings/...` → DB `AudioRecording(status=pending)` → Celery task queued → Worker picks up and writes `AnalysisResult` → Frontend polls for completion.

            Status lifecycle for a recording
            - `pending` → `processing` → `completed` / `failed`

            Dev setup (quick)
            1. Create and activate venv: `python -m venv venv` then `venv\\\\Scripts\\\\activate` (Windows).
            2. Install deps: `pip install -r requirements.txt`.
            3. Migrate DB: `python manage.py migrate`.
            4. Start Redis (or run via Docker). Example (Docker): `docker run -p 6379:6379 redis:latest`.
            5. Start Celery worker: `celery -A slaq_project worker --pool=solo --loglevel=info` (Windows dev) or without `--pool=solo` on Linux.
            6. Run server: `python manage.py runserver`.

            Important runtime environment variables
            - `REDIS_URL` — Redis connection URI
            - `DATABASE_URL` — database connection
            - `MODEL_DEVICE` — preferred device for model loading (`cpu`, `cuda`)
            - `MEDIA_ROOT` / `MEDIA_URL` — media storage

            API endpoints of interest
            - `GET /record/` — page to record audio
            - `POST /recordings/upload/` — upload handler
            - `GET /analyses/<id>/` — analysis page
            - `GET /api/analyses/<id>/status/` — returns `{status: 'pending'|'processing'|'completed'|'failed'}`
            - `GET /api/analyses/<id>/data/` — returns full analysis payload JSON

            Operational notes
            - Use batch workers to process queued tasks in production. Autoscale workers depending on queue length.
            - Keep media storage durable (S3) in production; ensure workers can access S3 or shared storage.
            - Use GPU workers for heavy inference; use CPU workers for light postprocessing tasks.

            Testing & validation
            - Unit tests: focus on `diagnosis/ai_engine/utils.py`, `detect_stuttering.py` and `tasks.py` orchestration.
            - Integration tests: simulate upload → queue → worker (mock model to avoid long runs).
            - Model tests: smoke test model loading and inference with short audio fixtures.

            Security and data privacy
            - Limit upload size and validate MIME type before saving.
            - Ensure patient data is access-controlled; only owners or assigned clinicians can view results.
            - Consider encrypting S3 buckets and backups containing audio.

            Where to look in the codebase
            - `diagnosis/ai_engine/` — model loader, detector, utils
            - `diagnosis/tasks.py` — Celery orchestration
            - `diagnosis/views.py` / `diagnosis/serializers.py` — endpoints and payloads
            - `core/models.py`, `diagnosis/models.py`, `reports/models.py` — DB models and relations

            Roadmap notes (short)
            - Improve timestamp precision: incorporate forced-alignment tools (e.g., Montreal Forced Aligner) or fine-tune models with phonetically annotated data.
            - Add confidence-calibrated severity scoring and clinician-facing edit tools for label correction.
            - Implement async websocket notifications for real-time status updates instead of polling.

            Contact & maintenance
            - For ML-related work: inspect `diagnosis/ai_engine/model_loader.py` and `diagnosis/ai_engine/detect_stuttering.py`.
            - For APIs and views: inspect `diagnosis/views.py` and `reports/` for report templates.

            Last updated: 2025-11-20
    - Calculates improvement scores



## 📊 Key Metrics Tracked

