'use client'

import { useState, useEffect } from 'react'
import { useOpenFaceAnalysis } from '@/hooks/useOpenFaceAnalysis'
import { useVideoManager } from '@/hooks/useVideoManager'
import { StatusDisplay } from '@/components/StatusDisplay'
import { ConnectionControls } from '@/components/ConnectionControls'
import { AnalysisControls } from '@/components/AnalysisControls'
import { VideoDisplay } from '@/components/VideoDisplay'
import { AnalysisResults } from '@/components/AnalysisResults'
import LogConsole from '@/components/LogConsoleNew'

export default function Home() {
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
    <div className="container">
      <div className="card">
        <div className="header">
          <h1>🎥 OpenFace-3.0 Real-time Analysis</h1>
          <p>Facial analysis with emotion detection and gaze tracking</p>
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

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          <VideoDisplay
            videoRef={video.videoRef}
            canvasRef={video.canvasRef}
            fps={analysis.fps}
          />
          <AnalysisResults results={analysis.analysisResults} />
        </div>

        {/* Real-time Logging Console */}
        <div className="mb-4" style={{ contain: 'layout' }}>
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
