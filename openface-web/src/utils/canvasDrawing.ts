import { AnalysisResult, Face } from '@/types'

export class CanvasDrawing {
  static drawVideoFrame(
    ctx: CanvasRenderingContext2D, 
    video: HTMLVideoElement, 
    canvas: HTMLCanvasElement
  ): void {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  }

  static drawAnalysisOverlay(
    ctx: CanvasRenderingContext2D, 
    data: AnalysisResult, 
    canvas: HTMLCanvasElement
  ): void {
    const scaleX = canvas.width / data.frame_info.width
    const scaleY = canvas.height / data.frame_info.height

    data.faces.forEach((face, index) => {
      this.drawFace(ctx, face, index, scaleX, scaleY)
    })
  }

  private static drawFace(
    ctx: CanvasRenderingContext2D, 
    face: Face, 
    index: number, 
    scaleX: number, 
    scaleY: number
  ): void {
    // Scale bounding box
    const bbox = [
      face.bbox[0] * scaleX,
      face.bbox[1] * scaleY,
      face.bbox[2] * scaleX,
      face.bbox[3] * scaleY
    ]

    // Draw bounding box
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])

    // Draw face label
    ctx.fillStyle = '#00ff00'
    ctx.font = '14px Arial'
    ctx.fillText(`Face ${index + 1} (${face.confidence.toFixed(2)})`, bbox[0], bbox[1] - 5)

    // Draw landmarks
    if (face.landmarks) {
      this.drawLandmarks(ctx, face.landmarks, scaleX, scaleY)
    }

    // Draw emotion
    if (face.emotion) {
      this.drawEmotion(ctx, face.emotion, bbox)
    }

    // Draw gaze arrow
    if (face.gaze) {
      this.drawGazeArrow(ctx, face.gaze, bbox)
    }

    // Draw Action Units
    if (face.action_units?.all_aus) {
      this.drawActionUnits(ctx, face.action_units.all_aus, bbox)
    }
  }

  private static drawLandmarks(
    ctx: CanvasRenderingContext2D, 
    landmarks: [number, number][], 
    scaleX: number, 
    scaleY: number
  ): void {
    ctx.fillStyle = '#ff0000'
    landmarks.forEach(point => {
      ctx.beginPath()
      ctx.arc(point[0] * scaleX, point[1] * scaleY, 2, 0, 2 * Math.PI)
      ctx.fill()
    })
  }

  private static drawEmotion(
    ctx: CanvasRenderingContext2D, 
    emotion: { label: string; confidence: number }, 
    bbox: number[]
  ): void {
    ctx.fillStyle = '#0066cc'
    ctx.font = '12px Arial'
    const emotionText = `${emotion.label}: ${(emotion.confidence * 100).toFixed(1)}%`
    ctx.fillText(emotionText, bbox[0], bbox[3] + 15)
  }

  private static drawGazeArrow(
    ctx: CanvasRenderingContext2D, 
    gaze: { pitch: number; yaw: number }, 
    bbox: number[]
  ): void {
    const centerX = (bbox[0] + bbox[2]) / 2
    const centerY = (bbox[1] + bbox[3]) / 2
    const arrowLength = 40
    const endX = centerX + gaze.yaw * arrowLength
    const endY = centerY + gaze.pitch * arrowLength

    ctx.strokeStyle = '#ff6600'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(centerX, centerY)
    ctx.lineTo(endX, endY)
    ctx.stroke()

    // Arrow head
    const angle = Math.atan2(endY - centerY, endX - centerX)
    ctx.beginPath()
    ctx.moveTo(endX, endY)
    ctx.lineTo(endX - 8 * Math.cos(angle - Math.PI/6), endY - 8 * Math.sin(angle - Math.PI/6))
    ctx.moveTo(endX, endY)
    ctx.lineTo(endX - 8 * Math.cos(angle + Math.PI/6), endY - 8 * Math.sin(angle + Math.PI/6))
    ctx.stroke()
  }

  private static drawActionUnits(
    ctx: CanvasRenderingContext2D, 
    actionUnits: Record<string, number>, 
    bbox: number[]
  ): void {
    const auEntries = Object.entries(actionUnits)
      .filter(([, intensity]) => intensity > 0.3)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)

    if (auEntries.length > 0) {
      ctx.fillStyle = '#ffcc00'
      ctx.font = '10px Arial'
      const auText = auEntries.map(([au, intensity]) => 
        `${au}:${Math.round(intensity * 100)}%`).join(', ')
      ctx.fillText(auText, bbox[0], bbox[3] + 30)
    }
  }
}
