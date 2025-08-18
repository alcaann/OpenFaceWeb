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
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastValidResultsRef = useRef<AnalysisResult | null>(null)
  const frameCountRef = useRef(0)
  const lastTimeRef = useRef(Date.now())

  // Update status helper
  const updateStatus = useCallback((message: string, statusType: ConnectionStatus) => {
    setStatusMessage(message)
    setStatus(statusType)
  }, [])

  // Initialize WebSocket service handlers
  useEffect(() => {
    const wsService = wsServiceRef.current
    
    wsService.setStatusChangeHandler(updateStatus)
    wsService.setAnalysisResultHandler((data: AnalysisResult) => {
      setIsProcessingFrame(false) // Frame processing completed
      
      if (data.success && data.faces && data.faces.length > 0) {
        lastValidResultsRef.current = data
        setAnalysisResults(data)
        console.log('✅ Analysis completed - ready for next frame')
      } else if (data.error) {
        console.error('Analysis error:', data.error)
        updateStatus('Analysis error: ' + data.error, 'disconnected')
      }
    })
  }, [updateStatus])

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
    stopAnalysis()
  }, [])

  // Frame capture and sending
  const captureAndSendFrame = useCallback((video: HTMLVideoElement, quality: number) => {
    if (!video.videoWidth || !wsServiceRef.current.isConnected() || isProcessingFrame) {
      if (isProcessingFrame) {
        console.log('⏸️ Skipping frame - previous analysis still processing')
      }
      return
    }

    setIsProcessingFrame(true)

    try {
      const imageData = VideoCapture.captureFrame(video, quality)
      wsServiceRef.current.sendFrame(imageData, Date.now())

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
      console.error('Frame capture failed:', error)
      setIsProcessingFrame(false)
    }
  }, [isProcessingFrame])

  // Analysis management
  const startAnalysis = useCallback((
    video: HTMLVideoElement, 
    frameRate: number, 
    videoQuality: number
  ) => {
    if (!wsServiceRef.current.isConnected()) {
      updateStatus('Cannot start analysis - check connection', 'disconnected')
      return
    }

    setIsAnalyzing(true)
    const analysisInterval_ms = 1000 / frameRate

    // Start analysis loop (sends frames to server)
    analysisIntervalRef.current = setInterval(() => {
      captureAndSendFrame(video, videoQuality)
    }, analysisInterval_ms)

    updateStatus(`Analysis started - ${frameRate} FPS`, 'processing')
  }, [captureAndSendFrame, updateStatus])

  const stopAnalysis = useCallback(() => {
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current)
      analysisIntervalRef.current = null
    }

    setIsAnalyzing(false)
    setIsProcessingFrame(false) // Reset processing flag

    updateStatus(
      wsServiceRef.current.isConnected() ? 'Connected - Analysis stopped' : 'Disconnected',
      wsServiceRef.current.isConnected() ? 'connected' : 'disconnected'
    )
  }, [updateStatus])

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
