import io, { Socket } from 'socket.io-client'
import { AnalysisResult, ConnectionStatus } from '@/types'

export class OpenFaceWebSocketService {
  private socket: Socket | null = null
  private onStatusChange: ((message: string, status: ConnectionStatus) => void) | null = null
  private onAnalysisResult: ((data: AnalysisResult) => void) | null = null
  private onReadyForNextFrame: ((timestamp: number) => void) | null = null

  constructor() {
    // Constructor can be empty or take initial configuration
  }

  connect(serverUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket) {
        this.socket.disconnect()
      }

      this.updateStatus('Connecting...', 'processing')
      console.log('🔌 Attempting to connect to:', serverUrl)

      this.socket = io(serverUrl, {
        transports: ['websocket', 'polling'],
        upgrade: true,
        rememberUpgrade: true
      })

      this.socket.on('connect', () => {
        console.log('✅ Connected to server!')
        this.updateStatus('Connected to OpenFace API', 'connected')
        resolve()
      })

      this.socket.on('disconnect', () => {
        console.log('❌ Disconnected from server')
        this.updateStatus('Disconnected from server', 'disconnected')
      })

      this.socket.on('connected', (data: any) => {
        console.log('📡 Server info:', data)
        this.updateStatus(`Connected - ${data.message}`, 'connected')
      })

      this.socket.on('analysis_result', (data: AnalysisResult) => {
        console.log('📊 Received analysis result:', data)
        if (this.onAnalysisResult) {
          this.onAnalysisResult(data)
        }
      })

      this.socket.on('ready_for_next_frame', (data: { timestamp: number }) => {
        console.log('🚀 Backend ready for next frame')
        if (this.onReadyForNextFrame) {
          this.onReadyForNextFrame(data.timestamp)
        }
      })

      this.socket.on('analysis_error', (data: { error: string }) => {
        console.error('❌ Analysis error:', data.error)
        this.updateStatus('Analysis error: ' + data.error, 'disconnected')
      })

      this.socket.on('connect_error', (error: any) => {
        console.error('❌ Connection error:', error)
        this.updateStatus('Connection failed: ' + error, 'disconnected')
        reject(error)
      })
    })
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    this.updateStatus('Disconnected', 'disconnected')
  }

  sendFrame(imageData: string, timestamp: number): void {
    if (!this.socket?.connected) {
      console.warn('⚠️ Cannot send frame - socket not connected')
      return
    }

    console.log('📤 Sending frame for analysis...', {
      timestamp,
      imageDataLength: imageData.length,
      socketConnected: this.socket.connected
    })
    this.socket.emit('analyze_frame', {
      image: imageData,
      timestamp
    })
  }

  isConnected(): boolean {
    return this.socket?.connected || false
  }

  getSocket(): Socket | null {
    return this.socket
  }

  setStatusChangeHandler(handler: (message: string, status: ConnectionStatus) => void): void {
    this.onStatusChange = handler
  }

  setAnalysisResultHandler(handler: (data: AnalysisResult) => void): void {
    this.onAnalysisResult = handler
  }

  setReadyForNextFrameHandler(handler: (timestamp: number) => void): void {
    this.onReadyForNextFrame = handler
  }

  private updateStatus(message: string, status: ConnectionStatus): void {
    if (this.onStatusChange) {
      this.onStatusChange(message, status)
    }
  }
}
