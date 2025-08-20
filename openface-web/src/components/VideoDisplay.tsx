import { RefObject } from 'react'

interface VideoDisplayProps {
  videoRef: RefObject<HTMLVideoElement>
  canvasRef: RefObject<HTMLCanvasElement>
  fps: number
}

export function VideoDisplay({ videoRef, canvasRef, fps }: VideoDisplayProps) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="mb-3 text-lg font-semibold text-gray-800">📹 Input Video</h3>
        <div className="relative inline-block">
          <video
            ref={videoRef}
            className="border-2 border-gray-300 rounded-lg max-w-full bg-black shadow-lg"
            width="640"
            height="480"
            autoPlay
            muted
          />
          <div className="absolute top-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs font-mono">
            FPS: {fps}
          </div>
        </div>
      </div>
      
      <div className="text-center">
        <h3 className="mb-3 text-lg font-semibold text-gray-800">🎯 Analysis Overlay</h3>
        <canvas
          ref={canvasRef}
          className="border-2 border-gray-300 rounded-lg max-w-full bg-black shadow-lg"
          width="640"
          height="480"
        />
      </div>
    </div>
  )
}
