export class VideoCapture {
  static async getUserMedia(constraints?: MediaStreamConstraints): Promise<MediaStream> {
    const defaultConstraints: MediaStreamConstraints = {
      video: {
        width: 640,
        height: 480,
        frameRate: 30
      }
    }

    return await navigator.mediaDevices.getUserMedia(constraints || defaultConstraints)
  }

  static captureFrame(video: HTMLVideoElement, quality: number = 0.8): string {
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = video.videoWidth
    tempCanvas.height = video.videoHeight
    const tempCtx = tempCanvas.getContext('2d')

    if (!tempCtx) {
      throw new Error('Unable to get 2D context from canvas')
    }

    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height)
    return tempCanvas.toDataURL('image/jpeg', quality)
  }

  static stopAllTracks(stream: MediaStream): void {
    stream.getTracks().forEach(track => track.stop())
  }
}
