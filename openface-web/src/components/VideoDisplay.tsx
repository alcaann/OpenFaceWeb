import { RefObject } from 'react'

interface VideoDisplayProps {
  videoRef: RefObject<HTMLVideoElement>
  canvasRef: RefObject<HTMLCanvasElement>
  fps: number
}

export function VideoDisplay({ videoRef, canvasRef, fps }: VideoDisplayProps) {
  return (
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
  )
}
