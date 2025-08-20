interface ConnectionControlsProps {
  serverUrl: string
  onServerUrlChange: (url: string) => void
  onConnect: () => void
  onDisconnect: () => void
  isConnected: boolean
}

export function ConnectionControls({
  serverUrl,
  onServerUrlChange,
  onConnect,
  onDisconnect,
  isConnected
}: ConnectionControlsProps) {
  return (
    <div className="my-5 p-4 bg-gray-50 rounded-lg">
      <h3 className="mb-4 text-lg font-semibold text-gray-800">Connection Settings</h3>
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
        <label className="min-w-[100px] font-medium text-gray-700">Server URL:</label>
        <input
          type="text"
          value={serverUrl}
          onChange={(e) => onServerUrlChange(e.target.value)}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="http://localhost:5000"
        />
        <div className="flex gap-2">
          <button 
            onClick={onConnect} 
            disabled={isConnected}
            className="px-4 py-2 bg-blue-600 text-white border-none rounded-md cursor-pointer text-sm font-medium transition-colors hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Connect
          </button>
          <button 
            onClick={onDisconnect} 
            disabled={!isConnected}
            className="px-4 py-2 bg-red-600 text-white border-none rounded-md cursor-pointer text-sm font-medium transition-colors hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Disconnect
          </button>
        </div>
      </div>
    </div>
  )
}
