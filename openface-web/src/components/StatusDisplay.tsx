import { ConnectionStatus } from '@/types'

interface StatusDisplayProps {
  status: ConnectionStatus
  message: string
}

export function StatusDisplay({ status, message }: StatusDisplayProps) {
  return (
    <div className={`status ${status}`}>
      {message}
    </div>
  )
}
