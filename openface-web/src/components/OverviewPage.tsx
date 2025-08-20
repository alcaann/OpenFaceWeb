'use client'

import { useState, useEffect } from 'react'
import { useOpenFaceAnalysis } from '@/hooks/useOpenFaceAnalysis'
import { useVideoManager } from '@/hooks/useVideoManager'
import { StatusDisplay } from '@/components/StatusDisplay'
import { ConnectionControls } from '@/components/ConnectionControls'
import { AnalysisControls } from '@/components/AnalysisControls'
import { VideoDisplay } from '@/components/VideoDisplay'
import { AnalysisResults } from '@/components/AnalysisResults'
import LogConsole from '@/components/LogConsole'

export function OverviewPage() {
  // Settings state
  const [serverUrl, setServerUrl] = useState('http://localhost:5000')
  const [frameRate, setFrameRate] = useState(10)
  const [videoQuality, setVideoQuality] = useState(0.8)

  // Custom hooks
  const analysis = useOpenFaceAnalysis()
  const video = useVideoManager()

  // Start continuous video rendering when camera is active
  useEffect(() => {
    if (video.isCameraActive) {
      video.startRenderLoop(analysis.lastValidResults)
    } else {
      video.stopRenderLoop()
    }
  }, [video.isCameraActive, analysis.lastValidResults, video])

  // Handlers
  const handleConnect = async () => {
    await analysis.connectToServer(serverUrl)
  }

  const handleStartVideo = async () => {
    try {
      await video.startVideo()
    } catch (error) {
      console.error('Failed to start video:', error)
    }
  }

  const handleStartAnalysis = () => {
    if (video.videoRef.current) {
      analysis.startAnalysis(video.videoRef.current, frameRate, videoQuality)
    }
  }

  const canStartAnalysis = analysis.isConnected() && video.isCameraActive

  return (
    <div className="max-w-7xl mx-auto p-5">
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg mb-5">
        <div className="text-center text-gray-800 dark:text-gray-200 mb-8">
          <h1 className="text-4xl font-bold mb-3">🎥 Real-time Facial Analysis</h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">Emotion detection and gaze tracking with OpenFace-3.0</p>
        </div>

        <StatusDisplay status={analysis.status} message={analysis.statusMessage} />

        <ConnectionControls
          serverUrl={serverUrl}
          onServerUrlChange={setServerUrl}
          onConnect={handleConnect}
          onDisconnect={analysis.disconnectFromServer}
          isConnected={analysis.isConnected()}
        />

        <AnalysisControls
          frameRate={frameRate}
          onFrameRateChange={setFrameRate}
          videoQuality={videoQuality}
          onVideoQualityChange={setVideoQuality}
          onStartVideo={handleStartVideo}
          onStopVideo={video.stopVideo}
          onStartAnalysis={handleStartAnalysis}
          onStopAnalysis={analysis.stopAnalysis}
          onTakeScreenshot={video.takeScreenshot}
          canStartAnalysis={canStartAnalysis}
          isAnalyzing={analysis.isAnalyzing}
          isCameraActive={video.isCameraActive}
          isStartingCamera={video.isStartingCamera}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <VideoDisplay
            videoRef={video.videoRef}
            canvasRef={video.canvasRef}
            fps={analysis.fps}
          />
          <AnalysisResults results={analysis.analysisResults} />
        </div>

        {/* Real-time Logging Console */}
        <div className="mb-6">
          <LogConsole 
            socket={analysis.socket} 
            maxLogs={100} 
            height="400px" 
          />
        </div>
      </div>
    </div>
  )
}
