import { useState, useCallback, useRef, useEffect } from 'react'
import { VideoCapture } from '@/utils/videoCapture'
import { CanvasDrawing } from '@/utils/canvasDrawing'
import { AnalysisResult } from '@/types'

export function useVideoManager() {
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [isStartingCamera, setIsStartingCamera] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const renderLoopRef = useRef<number | null>(null)

  const startVideo = useCallback(async () => {
    try {
      setIsStartingCamera(true)
      const stream = await VideoCapture.getUserMedia()

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        
        const setupCanvas = () => {
          if (canvasRef.current && videoRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth
            canvasRef.current.height = videoRef.current.videoHeight
            setIsCameraActive(true)
            setIsStartingCamera(false)
          }
        }
        
        // Set up the callback before playing
        videoRef.current.onloadedmetadata = setupCanvas
        
        await videoRef.current.play()
        
        // If metadata is already loaded, call setupCanvas immediately
        if (videoRef.current.readyState >= 1) {
          setupCanvas()
        }
      }
    } catch (error) {
      setIsStartingCamera(false)
      console.error('Camera access failed:', error)
      throw new Error(`Camera access failed: ${(error as Error).message}`)
    }
  }, [])

  const stopVideo = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      VideoCapture.stopAllTracks(stream)
      videoRef.current.srcObject = null
    }
    
    if (renderLoopRef.current) {
      cancelAnimationFrame(renderLoopRef.current)
      renderLoopRef.current = null
    }
    
    setIsCameraActive(false)
    setIsStartingCamera(false)
  }, [])

  const startRenderLoop = useCallback((lastValidResults: AnalysisResult | null) => {
    if (!videoRef.current || !canvasRef.current) return

    const renderFrame = () => {
      if (videoRef.current && canvasRef.current && !videoRef.current.paused && !videoRef.current.ended) {
        const ctx = canvasRef.current.getContext('2d')
        if (ctx) {
          // Draw video frame
          CanvasDrawing.drawVideoFrame(ctx, videoRef.current, canvasRef.current)
          
          // Draw analysis overlay if available
          if (lastValidResults?.success && lastValidResults.faces.length > 0) {
            CanvasDrawing.drawAnalysisOverlay(ctx, lastValidResults, canvasRef.current)
          }
        }
        renderLoopRef.current = requestAnimationFrame(renderFrame)
      }
    }
    
    renderFrame()
  }, [])

  const stopRenderLoop = useCallback(() => {
    if (renderLoopRef.current) {
      cancelAnimationFrame(renderLoopRef.current)
      renderLoopRef.current = null
    }
  }, [])

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
      stopVideo()
    }
  }, [stopVideo])

  return {
    // State
    isCameraActive,
    isStartingCamera,
    
    // Refs
    videoRef,
    canvasRef,
    
    // Actions
    startVideo,
    stopVideo,
    startRenderLoop,
    stopRenderLoop,
    takeScreenshot
  }
}
