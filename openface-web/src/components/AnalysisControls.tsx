interface AnalysisControlsProps {
  frameRate: number
  onFrameRateChange: (rate: number) => void
  videoQuality: number
  onVideoQualityChange: (quality: number) => void
  onStartVideo: () => void
  onStopVideo: () => void
  onStartAnalysis: () => void
  onStopAnalysis: () => void
  onTakeScreenshot: () => void
  canStartAnalysis: boolean
  isAnalyzing: boolean
  isCameraActive: boolean
  isStartingCamera?: boolean
}

export function AnalysisControls({
  frameRate,
  onFrameRateChange,
  videoQuality,
  onVideoQualityChange,
  onStartVideo,
  onStopVideo,
  onStartAnalysis,
  onStopAnalysis,
  onTakeScreenshot,
  canStartAnalysis,
  isAnalyzing,
  isCameraActive,
  isStartingCamera = false
}: AnalysisControlsProps) {
  return (
    <>
      <div className="settings">
        <h3>Analysis Settings</h3>
        <div className="setting-item">
          <label>Frame Rate:</label>
          <input
            type="range"
            min="1"
            max="30"
            value={frameRate}
            onChange={(e) => onFrameRateChange(parseInt(e.target.value))}
          />
          <span>{frameRate}</span> FPS
        </div>
        <div className="setting-item">
          <label>Video Quality:</label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={videoQuality}
            onChange={(e) => onVideoQualityChange(parseFloat(e.target.value))}
          />
          <span>{videoQuality}</span>
        </div>
      </div>

      <div className="controls">
        <button onClick={onStartVideo} disabled={isCameraActive || isStartingCamera}>
          {isStartingCamera ? 'Starting Camera...' : 'Start Camera'}
        </button>
        <button onClick={onStopVideo} disabled={!isCameraActive}>
          Stop Camera
        </button>
        <button onClick={onStartAnalysis} disabled={!canStartAnalysis || isAnalyzing}>
          Start Analysis
        </button>
        <button onClick={onStopAnalysis} disabled={!isAnalyzing}>
          Stop Analysis
        </button>
        <button onClick={onTakeScreenshot}>
          Take Screenshot
        </button>
      </div>
    </>
  )
}
