'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import io, { Socket } from 'socket.io-client'

interface Face {
  bbox: [number, number, number, number]
  confidence: number
  landmarks?: [number, number][]
  emotion?: {
    label: string
    confidence: number
  }
  gaze?: {
    pitch: number
    yaw: number
  }
  action_units?: {
    all_aus: Record<string, number>
  }
}

interface AnalysisResult {
  success: boolean
  faces: Face[]
  frame_info: {
    width: number
    height: number
    timestamp: number
  }
  error?: string
}

type ConnectionStatus = 'disconnected' | 'connected' | 'processing'

export default function Home() {
  // State management
  const [socket, setSocket] = useState<Socket | null>(null)
  const [status, setStatus] = useState<ConnectionStatus>('disconnected')
  const [statusMessage, setStatusMessage] = useState('Disconnected - Click Connect to start')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [serverUrl, setServerUrl] = useState('http://localhost:5000')
  const [frameRate, setFrameRate] = useState(10)
  const [videoQuality, setVideoQuality] = useState(0.8)
  const [fps, setFps] = useState(0)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null)
  const [isProcessingFrame, setIsProcessingFrame] = useState(false)

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const renderIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const frameCountRef = useRef(0)
  const lastTimeRef = useRef(Date.now())
  const lastValidResultsRef = useRef<AnalysisResult | null>(null)

  // Update status helper
  const updateStatus = useCallback((message: string, statusType: ConnectionStatus) => {
    setStatusMessage(message)
    setStatus(statusType)
  }, [])

  // Socket connection management
  const connectToServer = useCallback(() => {
    if (socket) {
      socket.disconnect()
    }

    updateStatus('Connecting...', 'processing')
    console.log('🔌 Attempting to connect to:', serverUrl)

    const newSocket = io(serverUrl, {
      transports: ['websocket', 'polling'],
      upgrade: true,
      rememberUpgrade: true
    })

    newSocket.on('connect', () => {
      console.log('✅ Connected to server!')
      updateStatus('Connected to OpenFace API', 'connected')
    })

    newSocket.on('disconnect', () => {
      console.log('❌ Disconnected from server')
      updateStatus('Disconnected from server', 'disconnected')
      setIsAnalyzing(false)
    })

    newSocket.on('connected', (data: any) => {
      console.log('📡 Server info:', data)
      updateStatus(`Connected - ${data.message}`, 'connected')
    })

    newSocket.on('analysis_result', (data: AnalysisResult) => {
      console.log('📊 Received analysis result:', data)
      handleAnalysisResult(data)
    })

    newSocket.on('analysis_error', (data: { error: string }) => {
      console.error('❌ Analysis error:', data.error)
      updateStatus('Analysis error: ' + data.error, 'disconnected')
    })

    newSocket.on('connect_error', (error: any) => {
      console.error('❌ Connection error:', error)
      updateStatus('Connection failed: ' + error, 'disconnected')
    })

    setSocket(newSocket)
  }, [serverUrl, socket, updateStatus])

  const disconnectFromServer = useCallback(() => {
    if (socket) {
      socket.disconnect()
      setSocket(null)
    }
    stopAnalysis()
    updateStatus('Disconnected', 'disconnected')
  }, [socket])

  // Draw analysis overlay function
  const drawAnalysisOverlay = useCallback((ctx: CanvasRenderingContext2D, data: AnalysisResult) => {
    if (!canvasRef.current) return

    const scaleX = canvasRef.current.width / data.frame_info.width
    const scaleY = canvasRef.current.height / data.frame_info.height

    data.faces.forEach((face, index) => {
      // Scale bounding box
      const bbox = [
        face.bbox[0] * scaleX,
        face.bbox[1] * scaleY,
        face.bbox[2] * scaleX,
        face.bbox[3] * scaleY
      ]

      // Draw bounding box
      ctx.strokeStyle = '#00ff00'
      ctx.lineWidth = 2
      ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])

      // Draw face label
      ctx.fillStyle = '#00ff00'
      ctx.font = '14px Arial'
      ctx.fillText(`Face ${index + 1} (${face.confidence.toFixed(2)})`, bbox[0], bbox[1] - 5)

      // Draw landmarks
      if (face.landmarks) {
        ctx.fillStyle = '#ff0000'
        face.landmarks.forEach(point => {
          ctx.beginPath()
          ctx.arc(point[0] * scaleX, point[1] * scaleY, 2, 0, 2 * Math.PI)
          ctx.fill()
        })
      }

      // Draw emotion
      if (face.emotion) {
        ctx.fillStyle = '#0066cc'
        ctx.font = '12px Arial'
        const emotionText = `${face.emotion.label}: ${(face.emotion.confidence * 100).toFixed(1)}%`
        ctx.fillText(emotionText, bbox[0], bbox[3] + 15)
      }

      // Draw gaze arrow
      if (face.gaze) {
        const centerX = (bbox[0] + bbox[2]) / 2
        const centerY = (bbox[1] + bbox[3]) / 2
        const arrowLength = 40
        const endX = centerX + face.gaze.yaw * arrowLength
        const endY = centerY + face.gaze.pitch * arrowLength

        ctx.strokeStyle = '#ff6600'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(endX, endY)
        ctx.stroke()

        // Arrow head
        const angle = Math.atan2(endY - centerY, endX - centerX)
        ctx.beginPath()
        ctx.moveTo(endX, endY)
        ctx.lineTo(endX - 8 * Math.cos(angle - Math.PI/6), endY - 8 * Math.sin(angle - Math.PI/6))
        ctx.moveTo(endX, endY)
        ctx.lineTo(endX - 8 * Math.cos(angle + Math.PI/6), endY - 8 * Math.sin(angle + Math.PI/6))
        ctx.stroke()
      }

      // Draw top Action Units
      if (face.action_units?.all_aus) {
        const auEntries = Object.entries(face.action_units.all_aus)
          .filter(([, intensity]) => intensity > 0.3)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 3)

        if (auEntries.length > 0) {
          ctx.fillStyle = '#ffcc00'
          ctx.font = '10px Arial'
          const auText = auEntries.map(([au, intensity]) => 
            `${au}:${Math.round(intensity * 100)}%`).join(', ')
          ctx.fillText(auText, bbox[0], bbox[3] + 30)
        }
      }
    })
  }, [])

  // Video management
  const startVideo = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          frameRate: 30
        }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        
        videoRef.current.onloadedmetadata = () => {
          if (canvasRef.current && videoRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth
            canvasRef.current.height = videoRef.current.videoHeight
            // Start continuous video rendering
            const drawVideoLoop = () => {
              if (videoRef.current && canvasRef.current && !videoRef.current.paused && !videoRef.current.ended) {
                const ctx = canvasRef.current.getContext('2d')
                if (ctx) {
                  ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
                  ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)
                  
                  // Draw analysis overlay if we have valid results
                  if (lastValidResultsRef.current?.success && lastValidResultsRef.current.faces.length > 0) {
                    drawAnalysisOverlay(ctx, lastValidResultsRef.current)
                  }
                }
                requestAnimationFrame(drawVideoLoop)
              }
            }
            drawVideoLoop()
          }
        }
        
        setIsCameraActive(true)
        updateStatus('Camera started', status === 'connected' ? 'connected' : 'disconnected')
      }
    } catch (error) {
      console.error('Camera access failed:', error)
      updateStatus(`Camera access failed: ${(error as Error).message}`, 'disconnected')
    }
  }, [updateStatus, status])

  const stopVideo = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setIsCameraActive(false)
    // Stop analysis if it's running
    if (isAnalyzing) {
      setIsAnalyzing(false)
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current)
        analysisIntervalRef.current = null
      }
      if (renderIntervalRef.current) {
        clearInterval(renderIntervalRef.current)
        renderIntervalRef.current = null
      }
    }
  }, [isAnalyzing])

  // Analysis management
  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !videoRef.current.videoWidth || !socket?.connected || isProcessingFrame) {
      if (isProcessingFrame) {
        console.log('⏸️ Skipping frame - previous analysis still processing')
      }
      return
    }

    setIsProcessingFrame(true)

    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = videoRef.current.videoWidth
    tempCanvas.height = videoRef.current.videoHeight
    const tempCtx = tempCanvas.getContext('2d')

    if (!tempCtx) return

    tempCtx.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height)
    const imageData = tempCanvas.toDataURL('image/jpeg', videoQuality)

    console.log('📤 Sending frame for analysis...')
    socket.emit('analyze_frame', {
      image: imageData,
      timestamp: Date.now()
    })

    // Update FPS counter
    frameCountRef.current++
    const now = Date.now()
    if (now - lastTimeRef.current >= 1000) {
      const currentFps = frameCountRef.current / ((now - lastTimeRef.current) / 1000)
      setFps(Math.round(currentFps * 10) / 10)
      frameCountRef.current = 0
      lastTimeRef.current = now
    }
  }, [socket, videoQuality, isProcessingFrame])

  const renderFrame = useCallback(() => {
    if (!videoRef.current || !videoRef.current.videoWidth || !canvasRef.current) {
      return
    }

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    // Always draw the video frame first
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)

    // Draw analysis overlay if we have valid results (regardless of analyzing state)
    if (lastValidResultsRef.current?.success && lastValidResultsRef.current.faces.length > 0) {
      drawAnalysisOverlay(ctx, lastValidResultsRef.current)
    }
  }, [])

  const handleAnalysisResult = useCallback((data: AnalysisResult) => {
    setIsProcessingFrame(false) // Frame processing completed
    
    if (data.success && data.faces && data.faces.length > 0) {
      lastValidResultsRef.current = data
      setAnalysisResults(data)
      console.log('✅ Analysis completed - ready for next frame')
    } else if (data.error) {
      console.error('Analysis error:', data.error)
      updateStatus('Analysis error: ' + data.error, 'disconnected')
    }
  }, [updateStatus])

  const startAnalysis = useCallback(() => {
    if (!socket?.connected || !isCameraActive) {
      updateStatus('Cannot start analysis - check connection and camera', 'disconnected')
      return
    }

    setIsAnalyzing(true)
    const analysisInterval_ms = 1000 / frameRate

    // Start analysis loop (sends frames to server)
    analysisIntervalRef.current = setInterval(captureAndSendFrame, analysisInterval_ms)

    // Start rendering loop (continuous smooth rendering at 30fps)
    renderIntervalRef.current = setInterval(renderFrame, 1000 / 30)

    updateStatus(`Analysis started - ${frameRate} FPS`, 'processing')
  }, [socket, isCameraActive, frameRate, captureAndSendFrame, renderFrame, updateStatus])

  const stopAnalysis = useCallback(() => {
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current)
      analysisIntervalRef.current = null
    }

    if (renderIntervalRef.current) {
      clearInterval(renderIntervalRef.current)
      renderIntervalRef.current = null
    }

    setIsAnalyzing(false)
    setIsProcessingFrame(false) // Reset processing flag
    // Don't clear lastValidResultsRef.current - keep the last overlay visible
    // setAnalysisResults(null) - also keep this for the UI display

    updateStatus(
      socket?.connected ? 'Connected - Analysis stopped' : 'Disconnected',
      socket?.connected ? 'connected' : 'disconnected'
    )
  }, [socket, updateStatus])

  const takeScreenshot = useCallback(() => {
    if (!canvasRef.current) return

    const link = document.createElement('a')
    link.download = `openface-analysis-${Date.now()}.png`
    link.href = canvasRef.current.toDataURL()
    link.click()
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (socket) {
        socket.disconnect()
      }
      stopAnalysis()
      stopVideo()
    }
  }, [])

  return (
    <div className="container">
      <div className="card">
        <div className="header">
          <h1>🎥 OpenFace-3.0 Web Client</h1>
          <p>Real-time facial analysis via WebSocket</p>
        </div>

        <div className="settings">
          <h3>Connection Settings</h3>
          <div className="setting-item">
            <label>Server URL:</label>
            <input
              type="text"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              placeholder="http://localhost:5000"
            />
            <button onClick={connectToServer}>Connect</button>
            <button onClick={disconnectFromServer}>Disconnect</button>
          </div>
          <div className="setting-item">
            <label>Frame Rate:</label>
            <input
              type="range"
              min="1"
              max="30"
              value={frameRate}
              onChange={(e) => setFrameRate(parseInt(e.target.value))}
            />
            <span>{frameRate} FPS</span>
          </div>
          <div className="setting-item">
            <label>Video Quality:</label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={videoQuality}
              onChange={(e) => setVideoQuality(parseFloat(e.target.value))}
            />
            <span>{videoQuality}</span>
          </div>
        </div>

        <div className={`status ${status}`}>
          {statusMessage}
        </div>

        <div className="controls">
          <button onClick={startVideo}>Start Camera</button>
          <button onClick={stopVideo}>Stop Camera</button>
          <button
            onClick={startAnalysis}
            disabled={!socket?.connected || !isCameraActive || isAnalyzing}
          >
            Start Analysis
          </button>
          <button
            onClick={stopAnalysis}
            disabled={!isAnalyzing}
          >
            Stop Analysis
          </button>
          <button onClick={takeScreenshot}>Take Screenshot</button>
        </div>

        <div className="video-container">
          <div className="video-section">
            <h3>Input Video</h3>
            <div style={{ position: 'relative' }}>
              <video
                ref={videoRef}
                className="video-element"
                width="640"
                height="480"
                autoPlay
                muted
              />
              <div className="fps-counter">FPS: {fps}</div>
            </div>
          </div>
          <div className="video-section">
            <h3>Analysis Overlay</h3>
            <canvas
              ref={canvasRef}
              className="video-element"
              width="640"
              height="480"
            />
          </div>
        </div>

        {analysisResults && (
          <div className="results">
            <h3>Analysis Results</h3>
            <div className="info-grid">
              <div className="info-item">
                <strong>Timestamp:</strong>
                {new Date(analysisResults.frame_info.timestamp * 1000).toLocaleTimeString()}
              </div>
              <div className="info-item">
                <strong>Faces Detected:</strong>
                {analysisResults.faces.length}
              </div>
              <div className="info-item">
                <strong>Frame Size:</strong>
                {analysisResults.frame_info.width}x{analysisResults.frame_info.height}
              </div>
              <div className="info-item">
                <strong>Status:</strong>
                {analysisResults.success ? 'Success' : 'Error'}
              </div>
            </div>

            {analysisResults.faces.map((face, index) => (
              <div key={index} className="face-info">
                <h4>Face {index + 1}</h4>
                <div className="info-grid">
                  <div className="info-item">
                    <strong>Confidence:</strong>
                    {(face.confidence * 100).toFixed(1)}%
                  </div>
                  {face.emotion && (
                    <div className="info-item">
                      <strong>Emotion:</strong>
                      {face.emotion.label} ({(face.emotion.confidence * 100).toFixed(1)}%)
                    </div>
                  )}
                  {face.gaze && (
                    <div className="info-item">
                      <strong>Gaze:</strong>
                      Pitch: {face.gaze.pitch.toFixed(2)}, Yaw: {face.gaze.yaw.toFixed(2)}
                    </div>
                  )}
                  {face.action_units?.all_aus && (
                    <div className="info-item">
                      <strong>Active AUs:</strong>
                      {Object.entries(face.action_units.all_aus)
                        .filter(([, intensity]) => intensity > 0.3)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 5)
                        .map(([au, intensity]) => `${au}(${Math.round(intensity * 100)}%)`)
                        .join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
