import { ConnectionStatus } from '@/types'

interface StatusDisplayProps {
  status: ConnectionStatus
  message: string
}

const getStatusStyles = (status: ConnectionStatus) => {
  const baseClasses = "my-5 p-3 rounded-lg text-center font-medium border"
  
  switch (status) {
    case 'connected':
      return `${baseClasses} bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-300 border-green-200 dark:border-green-700`
    case 'disconnected':
      return `${baseClasses} bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 border-red-200 dark:border-red-700`
    case 'processing':
      return `${baseClasses} bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-300 border-yellow-200 dark:border-yellow-700`
    default:
      return `${baseClasses} bg-gray-50 dark:bg-gray-800 text-gray-800 dark:text-gray-300 border-gray-200 dark:border-gray-700`
  }
}

export function StatusDisplay({ status, message }: StatusDisplayProps) {
  return (
    <div className={getStatusStyles(status)}>
      {message}
    </div>
  )
}
