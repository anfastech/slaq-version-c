// Audio Recorder with Web Audio API and Waveform Visualization

let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animationId = null;
let recordingStartTime = null;
let timerInterval = null;
let recordedBlob = null;
let audioElement = null;

// Initialize Audio Recorder
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

// Start Recording
async function startRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        // Initialize AudioContext for waveform visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        
        // Initialize MediaRecorder
        const mimeType = getSupportedMimeType();
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, { type: mimeType });
            stream.getTracks().forEach(track => track.stop());
            stopWaveformVisualization();
        };
        
        // Start recording
        mediaRecorder.start();
        recordingStartTime = Date.now();
        
        // Update UI
        updateUIForRecording(true);
        
        // Start timer
        startTimer();
        
        // Start waveform visualization
        startWaveformVisualization();
        
        console.log('Recording started');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Could not access microphone. Please ensure you have granted microphone permissions.');
    }
}

// Stop Recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        stopTimer();
        updateUIForRecording(false);
        console.log('Recording stopped');
    }
}

// Play Recording
function playRecording() {
    if (recordedBlob) {
        if (audioElement && !audioElement.paused) {
            audioElement.pause();
            audioElement.currentTime = 0;
            document.getElementById('play-recording-btn').innerHTML = `
                <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
                </svg>
                <span>Play</span>
            `;
        } else {
            const audioUrl = URL.createObjectURL(recordedBlob);
            audioElement = new Audio(audioUrl);
            audioElement.play();
            
            document.getElementById('play-recording-btn').innerHTML = `
                <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z"/>
                </svg>
                <span>Pause</span>
            `;
            
            audioElement.onended = () => {
                document.getElementById('play-recording-btn').innerHTML = `
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
                    </svg>
                    <span>Play</span>
                `;
            };
        }
    }
}

// Upload Recording
async function uploadRecording() {
    if (!recordedBlob) {
        alert('No recording to upload');
        return;
    }
    
    const formData = new FormData();
    const filename = `recording_${Date.now()}.webm`;
    formData.append('audio_file', recordedBlob, filename);
    
    // Get CSRF token
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    
    try {
        // Show progress
        document.getElementById('upload-progress').classList.remove('hidden');
        document.getElementById('upload-recording-btn').disabled = true;
        
        const response = await fetch('/diagnosis/upload/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            }
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Update progress to 100%
            document.getElementById('upload-progress-bar').style.width = '100%';
            document.getElementById('upload-status-text').textContent = 'Upload complete! Processing...';
            
            // Start polling for status
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

// Poll Recording Status
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
    }, 2000); // Poll every 2 seconds
}

// Reset Recording
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
    
    // Clear waveform
    const canvas = document.getElementById('waveform');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Timer Functions
function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        document.getElementById('timer-display').textContent = 
            `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
    document.getElementById('timer-display').classList.remove('hidden');
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// Waveform Visualization
function startWaveformVisualization() {
    const canvas = document.getElementById('waveform');
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        animationId = requestAnimationFrame(draw);
        
        analyser.getByteTimeDomainData(dataArray);
        
        ctx.fillStyle = 'rgb(243, 244, 246)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgb(0, 144, 80)'; // brand-green
        ctx.beginPath();
        
        const sliceWidth = canvas.width / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            
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

// UI Update Functions
function updateUIForRecording(isRecording) {
    const startBtn = document.getElementById('start-recording-btn');
    const stopBtn = document.getElementById('stop-recording-btn');
    const playBtn = document.getElementById('play-recording-btn');
    const uploadBtn = document.getElementById('upload-recording-btn');
    const resetBtn = document.getElementById('reset-recording-btn');
    const statusText = document.getElementById('status-text');
    
    if (isRecording) {
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        playBtn.classList.add('hidden');
        uploadBtn.classList.add('hidden');
        resetBtn.classList.add('hidden');
        statusText.textContent = 'Recording...';
        document.querySelector('#status-display').classList.add('recording-active');
    } else {
        startBtn.classList.add('hidden');
        stopBtn.classList.add('hidden');
        playBtn.classList.remove('hidden');
        uploadBtn.classList.remove('hidden');
        resetBtn.classList.remove('hidden');
        statusText.textContent = 'Recording complete';
        document.querySelector('#status-display').classList.remove('recording-active');
    }
}

// Get Supported MIME Type
function getSupportedMimeType() {
    const types = [
        'audio/webm',
        'audio/webm;codecs=opus',
        'audio/ogg;codecs=opus',
        'audio/mp4'
    ];
    
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }
    
    return 'audio/webm'; // fallback
}

// File Upload Handler
function initFileUpload() {
    const form = document.getElementById('file-upload-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('audio-file-input');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            // Validate file size (10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert('File too large. Maximum size is 10MB');
                return;
            }
            
            const formData = new FormData(form);
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
            
            try {
                document.getElementById('upload-progress').classList.remove('hidden');
                form.querySelector('button[type=submit]').disabled = true;
                
                const response = await fetch('/diagnosis/upload/', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': csrfToken
                    }
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
                form.querySelector('button[type=submit]').disabled = false;
            }
        });
    }
}
