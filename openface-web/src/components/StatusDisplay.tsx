import { ConnectionStatus } from '@/types'

interface StatusDisplayProps {
  status: ConnectionStatus
  message: string
}

const getStatusStyles = (status: ConnectionStatus) => {
  const baseClasses = "my-5 p-3 rounded-lg text-center font-medium border"
  
  switch (status) {
    case 'connected':
      return `${baseClasses} bg-green-50 text-green-800 border-green-200`
    case 'disconnected':
      return `${baseClasses} bg-red-50 text-red-800 border-red-200`
    case 'processing':
      return `${baseClasses} bg-yellow-50 text-yellow-800 border-yellow-200`
    default:
      return `${baseClasses} bg-gray-50 text-gray-800 border-gray-200`
  }
}

export function StatusDisplay({ status, message }: StatusDisplayProps) {
  return (
    <div className={getStatusStyles(status)}>
      {message}
    </div>
  )
}
