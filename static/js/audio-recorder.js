// static/js/audio-recorder.js

let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animationId = null;
let recordingStartTime = null;
let timerInterval = null;
let recordedBlob = null;
let audioElement = null;
// Track the mime type used for recording
let currentMimeType = 'audio/webm'; 

function initAudioRecorder() {
    const startBtn = document.getElementById('start-recording-btn');
    const stopBtn = document.getElementById('stop-recording-btn');
    const playBtn = document.getElementById('play-recording-btn');
    const uploadBtn = document.getElementById('upload-recording-btn');
    const resetBtn = document.getElementById('reset-recording-btn');
    
    if (startBtn) startBtn.addEventListener('click', startRecording);
    if (stopBtn) stopBtn.addEventListener('click', stopRecording);
    if (playBtn) playBtn.addEventListener('click', playRecording);
    if (uploadBtn) uploadBtn.addEventListener('click', uploadRecording);
    if (resetBtn) resetBtn.addEventListener('click', resetRecording);
}

function getSupportedMimeType() {
    const types = [
        'audio/webm',
        'audio/webm;codecs=opus',
        'audio/ogg;codecs=opus',
        'audio/mp4', // Safari support
        'audio/mp4;codecs=mp4a.40.2'
    ];
    
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
            console.log("Using supported MIME type:", type);
            return type;
        }
    }
    return 'audio/webm'; // fallback
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        
        currentMimeType = getSupportedMimeType();
        mediaRecorder = new MediaRecorder(stream, { mimeType: currentMimeType });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, { type: currentMimeType });
            stream.getTracks().forEach(track => track.stop());
            stopWaveformVisualization();
        };
        
        mediaRecorder.start();
        recordingStartTime = Date.now();
        
        updateUIForRecording(true);
        startTimer();
        startWaveformVisualization();
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Could not access microphone. Please ensure you have granted microphone permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        stopTimer();
        updateUIForRecording(false);
    }
}

function playRecording() {
    if (recordedBlob) {
        if (audioElement && !audioElement.paused) {
            audioElement.pause();
            audioElement.currentTime = 0;
            updatePlayButton(false);
        } else {
            const audioUrl = URL.createObjectURL(recordedBlob);
            audioElement = new Audio(audioUrl);
            audioElement.play();
            updatePlayButton(true);
            
            audioElement.onended = () => updatePlayButton(false);
        }
    }
}

function updatePlayButton(isPlaying) {
    const btn = document.getElementById('play-recording-btn');
    if (isPlaying) {
        btn.innerHTML = `<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z"/></svg><span>Pause</span>`;
    } else {
        btn.innerHTML = `<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/></svg><span>Play</span>`;
    }
}

async function uploadRecording() {
    if (!recordedBlob) {
        alert('No recording to upload');
        return;
    }
    
    const formData = new FormData();
    
    // Determine correct extension based on MIME type
    let extension = 'webm';
    if (currentMimeType.includes('mp4')) {
        extension = 'm4a'; // .m4a is safer for audio-only mp4 containers
    } else if (currentMimeType.includes('ogg')) {
        extension = 'ogg';
    } else if (currentMimeType.includes('wav')) {
        extension = 'wav';
    }
    
    const filename = `recording_${Date.now()}.${extension}`;
    console.log("Uploading file:", filename, "Size:", recordedBlob.size, "Type:", currentMimeType);

    formData.append('audio_file', recordedBlob, filename);
    
    // Get selected language (defaults to english)
    const langSelect = document.getElementById('language-select');
    const language = langSelect ? langSelect.value : 'english';
    formData.append('language', language);

    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    
    try {
        document.getElementById('upload-progress').classList.remove('hidden');
        document.getElementById('upload-recording-btn').disabled = true;
        
        const response = await fetch('/diagnosis/upload/', {
            method: 'POST',
            body: formData,
            headers: { 'X-CSRFToken': csrfToken }
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            document.getElementById('upload-progress-bar').style.width = '100%';
            document.getElementById('upload-status-text').textContent = 'Upload complete! Processing...';
            pollRecordingStatus(data.recording_id);
        } else {
            throw new Error(data.error || 'Upload failed');
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed: ' + error.message);
        document.getElementById('upload-progress').classList.add('hidden');
        document.getElementById('upload-recording-btn').disabled = false;
    }
}

// ... (Keep the existing pollRecordingStatus, resetRecording, timer, and visualization functions) ...
// Only paste the functions above if you want to be minimal, but ideally replace the whole file to be safe.
// For brevity, I assume you can keep the rest of the file (pollRecordingStatus downwards) 
// or ask if you need the full file content again.

// --- Re-adding critical helper functions just in case ---

function pollRecordingStatus(recordingId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/diagnosis/api/status/${recordingId}/`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                document.getElementById('upload-status-text').textContent = 'Analysis complete!';
                setTimeout(() => {
                    window.location.href = `/diagnosis/analysis/${data.analysis_id}/`;
                }, 1500);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                document.getElementById('upload-status-text').textContent = 'Analysis failed: ' + data.error_message;
                document.getElementById('upload-recording-btn').disabled = false;
            } else if (data.status === 'processing') {
                document.getElementById('upload-status-text').textContent = 'Processing audio...';
            }
        } catch (error) {
            console.error('Status poll error:', error);
        }
    }, 2000);
}

function resetRecording() {
    if (audioElement) {
        audioElement.pause();
        audioElement = null;
    }
    recordedBlob = null;
    audioChunks = [];
    updateUIForRecording(false);
    document.getElementById('play-recording-btn').classList.add('hidden');
    document.getElementById('upload-recording-btn').classList.add('hidden');
    document.getElementById('reset-recording-btn').classList.add('hidden');
    document.getElementById('upload-progress').classList.add('hidden');
    document.getElementById('timer-display').classList.add('hidden');
    document.getElementById('timer-display').textContent = '00:00';
    document.getElementById('status-text').textContent = 'Ready to record';
    
    const canvas = document.getElementById('waveform');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        const display = document.getElementById('timer-display');
        if(display) display.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
    document.getElementById('timer-display').classList.remove('hidden');
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function startWaveformVisualization() {
    const canvas = document.getElementById('waveform');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        animationId = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);
        ctx.fillStyle = 'rgb(243, 244, 246)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgb(0, 144, 80)';
        ctx.beginPath();
        const sliceWidth = canvas.width / bufferLength;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
    draw();
}

function stopWaveformVisualization() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

function updateUIForRecording(isRecording) {
    const startBtn = document.getElementById('start-recording-btn');
    const stopBtn = document.getElementById('stop-recording-btn');
    const playBtn = document.getElementById('play-recording-btn');
    const uploadBtn = document.getElementById('upload-recording-btn');
    const resetBtn = document.getElementById('reset-recording-btn');
    const statusText = document.getElementById('status-text');
    const statusDisplay = document.getElementById('status-display');
    
    if (isRecording) {
        if(startBtn) startBtn.classList.add('hidden');
        if(stopBtn) stopBtn.classList.remove('hidden');
        if(playBtn) playBtn.classList.add('hidden');
        if(uploadBtn) uploadBtn.classList.add('hidden');
        if(resetBtn) resetBtn.classList.add('hidden');
        if(statusText) statusText.textContent = 'Recording...';
        if(statusDisplay) statusDisplay.classList.add('recording-active');
    } else {
        if(startBtn) startBtn.classList.add('hidden');
        if(stopBtn) stopBtn.classList.add('hidden');
        if(playBtn) playBtn.classList.remove('hidden');
        if(uploadBtn) uploadBtn.classList.remove('hidden');
        if(resetBtn) resetBtn.classList.remove('hidden');
        if(statusText) statusText.textContent = 'Recording complete';
        if(statusDisplay) statusDisplay.classList.remove('recording-active');
    }
}

function initFileUpload() {
    const form = document.getElementById('file-upload-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('audio-file-input');
            const file = fileInput.files[0];
            if (!file) { alert('Please select a file'); return; }
            if (file.size > 10 * 1024 * 1024) { alert('File too large. Maximum size is 10MB'); return; }
            
            const formData = new FormData(form);
            // Add language selection if present in the UI
            const langSelect = document.getElementById('language-select');
            if (langSelect) formData.append('language', langSelect.value);

            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
            try {
                document.getElementById('upload-progress').classList.remove('hidden');
                form.querySelector('button[type=submit]').disabled = true;
                const response = await fetch('/diagnosis/upload/', {
                    method: 'POST', body: formData, headers: { 'X-CSRFToken': csrfToken }
                });
                const data = await response.json();
                if (response.ok && data.success) {
                    document.getElementById('upload-progress-bar').style.width = '100%';
                    document.getElementById('upload-status-text').textContent = 'Upload complete! Processing...';
                    pollRecordingStatus(data.recording_id);
                } else { throw new Error(data.error || 'Upload failed'); }
            } catch (error) {
                console.error('Upload error:', error);
                alert('Upload failed: ' + error.message);
                document.getElementById('upload-progress').classList.add('hidden');
                form.querySelector('button[type=submit]').disabled = false;
            }
        });
    }
}