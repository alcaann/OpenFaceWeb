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
  const buttonBaseClasses = "px-4 py-2 text-white border-none rounded-md cursor-pointer text-sm font-medium transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
  
  return (
    <>
      <div className="my-5 p-4 bg-gray-50 rounded-lg">
        <h3 className="mb-4 text-lg font-semibold text-gray-800">Analysis Settings</h3>
        
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
            <label className="min-w-[120px] font-medium text-gray-700">Frame Rate:</label>
            <input
              type="range"
              min="1"
              max="30"
              value={frameRate}
              onChange={(e) => onFrameRateChange(parseInt(e.target.value))}
              className="flex-1 min-w-[150px] accent-blue-600"
            />
            <span className="text-sm font-mono bg-white px-2 py-1 rounded border">
              {frameRate} FPS
            </span>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
            <label className="min-w-[120px] font-medium text-gray-700">Video Quality:</label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={videoQuality}
              onChange={(e) => onVideoQualityChange(parseFloat(e.target.value))}
              className="flex-1 min-w-[150px] accent-blue-600"
            />
            <span className="text-sm font-mono bg-white px-2 py-1 rounded border">
              {videoQuality}
            </span>
          </div>
        </div>
      </div>

      <div className="my-5 text-center">
        <div className="flex flex-wrap gap-3 justify-center">
          <button 
            onClick={onStartVideo} 
            disabled={isCameraActive || isStartingCamera}
            className={`${buttonBaseClasses} bg-green-600 hover:bg-green-700`}
          >
            {isStartingCamera ? '🔄 Starting Camera...' : '📹 Start Camera'}
          </button>
          
          <button 
            onClick={onStopVideo} 
            disabled={!isCameraActive}
            className={`${buttonBaseClasses} bg-red-600 hover:bg-red-700`}
          >
            ⏹️ Stop Camera
          </button>
          
          <button 
            onClick={onStartAnalysis} 
            disabled={!canStartAnalysis || isAnalyzing}
            className={`${buttonBaseClasses} bg-blue-600 hover:bg-blue-700`}
          >
            🔍 Start Analysis
          </button>
          
          <button 
            onClick={onStopAnalysis} 
            disabled={!isAnalyzing}
            className={`${buttonBaseClasses} bg-orange-600 hover:bg-orange-700`}
          >
            ⏸️ Stop Analysis
          </button>
          
          <button 
            onClick={onTakeScreenshot}
            className={`${buttonBaseClasses} bg-purple-600 hover:bg-purple-700`}
          >
            📸 Screenshot
          </button>
        </div>
      </div>
    </>
  )
}
