export interface Face {
  bbox: [number, number, number, number]
  confidence: number
  landmarks?: [number, number][]
  emotion?: {
    label: string
    confidence: number
  }
  gaze?: {
    pitch: number
    yaw: number
  }
  action_units?: {
    all_aus: Record<string, number>
  }
}

export interface AnalysisResult {
  success: boolean
  faces: Face[]
  frame_info: {
    width: number
    height: number
    timestamp: number
  }
  error?: string
}

export type ConnectionStatus = 'disconnected' | 'connected' | 'processing'
