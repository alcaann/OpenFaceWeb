import { useState, useCallback, useRef, useEffect } from 'react'
import { AnalysisResult, ConnectionStatus } from '@/types'
import { OpenFaceWebSocketService } from '@/services/websocket'
import { VideoCapture } from '@/utils/videoCapture'

export function useOpenFaceAnalysis() {
  // State
  const [status, setStatus] = useState<ConnectionStatus>('disconnected')
  const [statusMessage, setStatusMessage] = useState('Disconnected - Click Connect to start')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isProcessingFrame, setIsProcessingFrame] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null)
  const [fps, setFps] = useState(0)

  // Refs
  const wsServiceRef = useRef<OpenFaceWebSocketService>(new OpenFaceWebSocketService())
  const lastValidResultsRef = useRef<AnalysisResult | null>(null)
  const frameCountRef = useRef(0)
  const lastTimeRef = useRef(Date.now())
  
  // New refs for latest frame strategy
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const videoQualityRef = useRef<number>(0.8)
  const latestFrameBufferRef = useRef<string | null>(null)
  const frameCaptureBusyRef = useRef(false)
  const continuousCaptureRef = useRef<number | null>(null)

  // Update status helper
  const updateStatus = useCallback((message: string, statusType: ConnectionStatus) => {
    setStatusMessage(message)
    setStatus(statusType)
  }, [])

  // New frame capture strategy functions
  const captureLatestFrame = useCallback(() => {
    if (!videoRef.current || !videoRef.current.videoWidth || frameCaptureBusyRef.current) {
      console.log('📸 Skipping frame capture - video not ready or busy')
      return
    }

    frameCaptureBusyRef.current = true
    try {
      const imageData = VideoCapture.captureFrame(videoRef.current, videoQualityRef.current)
      latestFrameBufferRef.current = imageData
      console.log('📸 Captured latest frame - buffer updated')
    } catch (error) {
      console.error('Frame capture failed:', error)
    } finally {
      frameCaptureBusyRef.current = false
    }
  }, [])

  const sendLatestFrame = useCallback(() => {
    if (!latestFrameBufferRef.current || !wsServiceRef.current.isConnected()) {
      console.log('⚠️ Cannot send frame - missing buffer or not connected')
      return
    }

    if (isProcessingFrame) {
      console.log('⚠️ Cannot send frame - still processing previous frame')
      return
    }

    console.log('📤 Sending latest frame to backend')
    setIsProcessingFrame(true)

    try {
      wsServiceRef.current.sendFrame(latestFrameBufferRef.current, Date.now())
      
      // Update FPS counter
      frameCountRef.current++
      const now = Date.now()
      if (now - lastTimeRef.current >= 1000) {
        const currentFps = frameCountRef.current / ((now - lastTimeRef.current) / 1000)
        setFps(Math.round(currentFps * 10) / 10)
        frameCountRef.current = 0
        lastTimeRef.current = now
      }
    } catch (error) {
      console.error('Frame sending failed:', error)
      setIsProcessingFrame(false)
    }
  }, [isProcessingFrame])

  const startContinuousCapture = useCallback(() => {
    console.log('🎬 Starting continuous frame capture')
    if (continuousCaptureRef.current) {
      cancelAnimationFrame(continuousCaptureRef.current)
    }

    const captureLoop = () => {
      captureLatestFrame()
      continuousCaptureRef.current = requestAnimationFrame(captureLoop)
    }
    captureLoop()
  }, [captureLatestFrame])

  const stopContinuousCapture = useCallback(() => {
    console.log('🛑 Stopping continuous frame capture')
    if (continuousCaptureRef.current) {
      cancelAnimationFrame(continuousCaptureRef.current)
      continuousCaptureRef.current = null
    }
  }, [])

  // Initialize WebSocket service handlers
  useEffect(() => {
    const wsService = wsServiceRef.current
    
    wsService.setStatusChangeHandler(updateStatus)
    wsService.setAnalysisResultHandler((data: AnalysisResult) => {
      // Don't reset isProcessingFrame here - wait for ready signal
      console.log('📊 Analysis result received:', {
        success: data.success,
        faces: data.faces?.length || 0,
        error: data.error
      })
      
      if (data.success && data.faces && data.faces.length > 0) {
        lastValidResultsRef.current = data
        setAnalysisResults(data)
        console.log('✅ Analysis completed - waiting for ready signal')
      } else if (data.error) {
        console.error('Analysis error:', data.error)
        updateStatus('Analysis error: ' + data.error, 'disconnected')
      }
    })

    // New handler for ready signal from backend
    wsService.setReadyForNextFrameHandler((timestamp: number) => {
      console.log('🚀 Backend ready signal received - resetting processing flag and sending latest frame')
      console.log('📋 Current state before processing ready signal:', {
        isProcessingFrame,
        hasLatestFrame: !!latestFrameBufferRef.current,
        isConnected: wsServiceRef.current.isConnected()
      })
      
      // Force reset processing flag and send frame
      setIsProcessingFrame(false)
      
      // Use a small timeout to ensure state update takes effect
      setTimeout(() => {
        console.log('📋 State after timeout:', {
          hasLatestFrame: !!latestFrameBufferRef.current,
          isConnected: wsServiceRef.current.isConnected()
        })
        
        if (latestFrameBufferRef.current && wsServiceRef.current.isConnected()) {
          console.log('📤 Sending next frame after ready signal')
          wsServiceRef.current.sendFrame(latestFrameBufferRef.current, Date.now())
          setIsProcessingFrame(true)
          
          // Update FPS counter
          frameCountRef.current++
          const now = Date.now()
          if (now - lastTimeRef.current >= 1000) {
            const currentFps = frameCountRef.current / ((now - lastTimeRef.current) / 1000)
            setFps(Math.round(currentFps * 10) / 10)
            frameCountRef.current = 0
            lastTimeRef.current = now
          }
        } else {
          console.log('⚠️ Cannot send frame - missing buffer or not connected')
        }
      }, 10)
    })
  }, [updateStatus, sendLatestFrame])

  // Connection management
  const connectToServer = useCallback(async (serverUrl: string) => {
    try {
      await wsServiceRef.current.connect(serverUrl)
    } catch (error) {
      console.error('Connection failed:', error)
    }
  }, [])

  const disconnectFromServer = useCallback(() => {
    wsServiceRef.current.disconnect()
    // stopAnalysis will be called by the cleanup effect
  }, [])

  // Analysis management
  const startAnalysis = useCallback((
    video: HTMLVideoElement, 
    frameRate: number, 
    videoQuality: number
  ) => {
    console.log('🚀 Starting analysis with params:', { frameRate, videoQuality })
    
    if (!wsServiceRef.current.isConnected()) {
      console.log('❌ Cannot start analysis - not connected')
      updateStatus('Cannot start analysis - check connection', 'disconnected')
      return
    }

    // Store video reference and quality for continuous capture
    videoRef.current = video
    videoQualityRef.current = videoQuality

    setIsAnalyzing(true)

    // Start continuous frame capture
    startContinuousCapture()

    // Send initial frame to start the process
    console.log('📸 Capturing initial frame...')
    captureLatestFrame()
    setTimeout(() => {
      console.log('📤 Sending initial frame...')
      sendLatestFrame()
    }, 100) // Small delay to ensure frame is captured

    updateStatus(`Analysis started - Latest frame mode`, 'processing')
  }, [startContinuousCapture, captureLatestFrame, sendLatestFrame, updateStatus])

  const stopAnalysis = useCallback(() => {
    // Stop continuous capture
    stopContinuousCapture()

    setIsAnalyzing(false)
    setIsProcessingFrame(false) // Reset processing flag
    
    // Clear references
    videoRef.current = null
    latestFrameBufferRef.current = null

    updateStatus(
      wsServiceRef.current.isConnected() ? 'Connected - Analysis stopped' : 'Disconnected',
      wsServiceRef.current.isConnected() ? 'connected' : 'disconnected'
    )
  }, [stopContinuousCapture, updateStatus])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsServiceRef.current.disconnect()
      stopAnalysis()
    }
  }, [stopAnalysis])

  return {
    // State
    status,
    statusMessage,
    isAnalyzing,
    isProcessingFrame,
    analysisResults,
    fps,
    lastValidResults: lastValidResultsRef.current,
    
    // Actions
    connectToServer,
    disconnectFromServer,
    startAnalysis,
    stopAnalysis,
    
    // Utils
    isConnected: () => wsServiceRef.current.isConnected(),
    
    // WebSocket access for logging
    socket: wsServiceRef.current.getSocket()
  }
}
