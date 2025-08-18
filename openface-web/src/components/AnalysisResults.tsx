import { AnalysisResult } from '@/types'

interface AnalysisResultsProps {
  results: AnalysisResult | null
}

export function AnalysisResults({ results }: AnalysisResultsProps) {
  if (!results) return null

  return (
    <div className="results">
      <h3>Analysis Results</h3>
      <div className="info-grid">
        <div className="info-item">
          <strong>Timestamp:</strong>
          {new Date(results.frame_info.timestamp * 1000).toLocaleTimeString()}
        </div>
        <div className="info-item">
          <strong>Faces Detected:</strong>
          {results.faces.length}
        </div>
        <div className="info-item">
          <strong>Frame Size:</strong>
          {results.frame_info.width}x{results.frame_info.height}
        </div>
        <div className="info-item">
          <strong>Status:</strong>
          {results.success ? 'Success' : 'Error'}
        </div>
      </div>

      {results.faces.map((face, index) => (
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
  )
}
