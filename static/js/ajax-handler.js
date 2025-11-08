// AJAX Handler Utilities for SLAQ

// Generic AJAX Request Handler
async function ajaxRequest(url, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        }
    };
    
    const config = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('AJAX Request Error:', error);
        throw error;
    }
}

// GET Request
async function ajaxGet(url) {
    return ajaxRequest(url, { method: 'GET' });
}

// POST Request
async function ajaxPost(url, data) {
    return ajaxRequest(url, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

// PUT Request
async function ajaxPut(url, data) {
    return ajaxRequest(url, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
}

// DELETE Request
async function ajaxDelete(url) {
    return ajaxRequest(url, { method: 'DELETE' });
}

// Form Data Upload (for files)
async function ajaxUpload(url, formData) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCSRFToken()
            }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `Upload failed! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Upload Error:', error);
        throw error;
    }
}

// Polling Function (for checking status)
function startPolling(url, callback, interval = 2000, maxAttempts = 60) {
    let attempts = 0;
    
    const pollInterval = setInterval(async () => {
        attempts++;
        
        try {
            const data = await ajaxGet(url);
            const shouldContinue = callback(data);
            
            if (!shouldContinue || attempts >= maxAttempts) {
                clearInterval(pollInterval);
            }
        } catch (error) {
            console.error('Polling error:', error);
            clearInterval(pollInterval);
        }
    }, interval);
    
    return pollInterval;
}

// Recording Status Poller
function pollRecordingStatus(recordingId, onUpdate, onComplete, onError) {
    return startPolling(
        `/diagnosis/api/status/${recordingId}/`,
        (data) => {
            if (onUpdate) onUpdate(data);
            
            if (data.status === 'completed') {
                if (onComplete) onComplete(data);
                return false; // Stop polling
            } else if (data.status === 'failed') {
                if (onError) onError(data);
                return false; // Stop polling
            }
            
            return true; // Continue polling
        },
        2000,
        60 // Max 2 minutes
    );
}

// Batch Request Handler
async function batchRequests(urls) {
    try {
        const promises = urls.map(url => ajaxGet(url));
        return await Promise.all(promises);
    } catch (error) {
        console.error('Batch request error:', error);
        throw error;
    }
}

// Retry Failed Requests
async function retryRequest(requestFn, maxRetries = 3, delay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await requestFn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
        }
    }
}

// Request with Timeout
async function requestWithTimeout(requestFn, timeout = 30000) {
    return Promise.race([
        requestFn(),
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Request timeout')), timeout)
        )
    ]);
}

// Download File via AJAX
async function downloadFile(url, filename) {
    try {
        const response = await fetch(url, {
            headers: {
                'X-CSRFToken': getCSRFToken()
            }
        });
        
        if (!response.ok) {
            throw new Error(`Download failed! status: ${response.status}`);
        }
        
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
        console.error('Download error:', error);
        throw error;
    }
}

// Helper to get CSRF Token
function getCSRFToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

// Loading Indicator
function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="flex justify-center items-center py-8">
                <div class="spinner"></div>
            </div>
        `;
    }
}

function hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
    }
}

// Error Handler
function handleAjaxError(error, showToastNotification = true) {
    console.error('AJAX Error:', error);
    
    if (showToastNotification && window.SLAQ && window.SLAQ.showToast) {
        window.SLAQ.showToast(error.message || 'An error occurred', 'error');
    }
    
    return error;
}

// Export functions
window.AJAX = {
    request: ajaxRequest,
    get: ajaxGet,
    post: ajaxPost,
    put: ajaxPut,
    delete: ajaxDelete,
    upload: ajaxUpload,
    poll: startPolling,
    pollRecordingStatus,
    batch: batchRequests,
    retry: retryRequest,
    withTimeout: requestWithTimeout,
    download: downloadFile,
    showLoading,
    hideLoading,
    handleError: handleAjaxError
};
