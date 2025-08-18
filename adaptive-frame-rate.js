// Add this to the test_client.html JavaScript section
// Adaptive frame rate based on backend response time

let lastSentTime = 0;
let lastReceivedTime = 0;
let adaptiveFrameRate = 10;
let targetFrameRate = 10;

function captureAndSendFrameAdaptive() {
    if (!video || !video.videoWidth || !socket || !socket.connected) {
        return;
    }
    
    const now = Date.now();
    lastSentTime = now;
    
    // Create a temporary canvas for capture
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw current video frame to temp canvas
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Convert to base64
    const quality = parseFloat(document.getElementById('videoQuality').value);
    const imageData = tempCanvas.toDataURL('image/jpeg', quality);
    
    // Send to server with timestamp
    socket.emit('analyze_frame', {
        image: imageData,
        timestamp: now,
        client_sent_time: now
    });
    
    // Update FPS counter
    frameCount++;
    if (now - lastTime >= 1000) {
        const fps = frameCount / ((now - lastTime) / 1000);
        document.getElementById('fpsCounter').textContent = `FPS: ${fps.toFixed(1)} (Target: ${adaptiveFrameRate})`;
        frameCount = 0;
        lastTime = now;
    }
}

function handleAnalysisResultAdaptive(data) {
    lastReceivedTime = Date.now();
    const processingTime = lastReceivedTime - data.client_sent_time;
    
    // Adapt frame rate based on processing time
    if (processingTime > 200) { // If backend takes > 200ms
        adaptiveFrameRate = Math.max(2, adaptiveFrameRate - 1);
    } else if (processingTime < 50) { // If backend is fast < 50ms
        adaptiveFrameRate = Math.min(targetFrameRate, adaptiveFrameRate + 1);
    }
    
    // Restart interval with new frame rate
    if (analysisInterval) {
        clearInterval(analysisInterval);
        analysisInterval = setInterval(captureAndSendFrameAdaptive, 1000 / adaptiveFrameRate);
    }
    
    currentResults = data;
    
    if (data.success && data.faces && data.faces.length > 0) {
        lastValidResults = data;
        updateResultsDisplay(data);
    } else if (data.error) {
        console.error('Analysis error:', data.error);
        updateStatus('Analysis error: ' + data.error, 'disconnected');
    }
}
