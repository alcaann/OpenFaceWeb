import { AnalysisResult } from '@/types'

interface AnalysisResultsProps {
  results: AnalysisResult | null
}

export function AnalysisResults({ results }: AnalysisResultsProps) {
  if (!results) {
    return (
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border-l-4 border-blue-500">
        <h3 className="mb-3 text-lg font-semibold text-gray-800 dark:text-gray-200">📊 Analysis Results</h3>
        <p className="text-gray-600 dark:text-gray-400">No analysis data available</p>
      </div>
    )
  }

  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border-l-4 border-blue-500">
      <h3 className="mb-4 text-lg font-semibold text-gray-800 dark:text-gray-200">📊 Analysis Results</h3>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        <div className="bg-white dark:bg-gray-700 p-3 rounded-md text-sm">
          <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">⏰ Timestamp</div>
          <div className="text-gray-600 dark:text-gray-400">
            {new Date(results.frame_info.timestamp * 1000).toLocaleTimeString()}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-700 p-3 rounded-md text-sm">
          <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">👤 Faces Detected</div>
          <div className="text-gray-600 dark:text-gray-400">{results.faces.length}</div>
        </div>
        
        <div className="bg-white dark:bg-gray-700 p-3 rounded-md text-sm">
          <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">📐 Frame Size</div>
          <div className="text-gray-600 dark:text-gray-400">
            {results.frame_info.width}x{results.frame_info.height}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-700 p-3 rounded-md text-sm">
          <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">✅ Status</div>
          <div className={`font-medium ${results.success ? 'text-green-600' : 'text-red-600'}`}>
            {results.success ? 'Success' : 'Error'}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {results.faces.map((face, index) => (
          <div key={index} className="bg-white dark:bg-gray-700 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
            <h4 className="mb-3 text-md font-semibold text-blue-600 dark:text-blue-400">
              👤 Face {index + 1}
            </h4>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="bg-gray-50 dark:bg-gray-600 p-3 rounded-md text-sm">
                <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">🎯 Confidence</div>
                <div className="text-gray-600 dark:text-gray-400">{(face.confidence * 100).toFixed(1)}%</div>
              </div>
              
              {face.emotion && (
                <div className="bg-gray-50 dark:bg-gray-600 p-3 rounded-md text-sm">
                  <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">😊 Emotion</div>
                  <div className="text-gray-600 dark:text-gray-400">
                    {face.emotion.label} ({(face.emotion.confidence * 100).toFixed(1)}%)
                  </div>
                </div>
              )}
              
              {face.gaze && (
                <div className="bg-gray-50 dark:bg-gray-600 p-3 rounded-md text-sm">
                  <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">👀 Gaze</div>
                  <div className="text-gray-600 dark:text-gray-400 text-xs font-mono">
                    Pitch: {face.gaze.pitch.toFixed(2)}<br />
                    Yaw: {face.gaze.yaw.toFixed(2)}
                  </div>
                </div>
              )}
              
              {face.action_units?.all_aus && (
                <div className="bg-gray-50 dark:bg-gray-600 p-3 rounded-md text-sm col-span-full">
                  <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">🎭 Active Action Units</div>
                  <div className="text-gray-600 dark:text-gray-400 text-xs">
                    {Object.entries(face.action_units.all_aus)
                      .filter(([, intensity]) => intensity > 0.3)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 5)
                      .map(([au, intensity]) => (
                        <span key={au} className="inline-block bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-2 py-1 rounded mr-1 mb-1">
                          {au}({Math.round(intensity * 100)}%)
                        </span>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
