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
    <div className="settings">
      <h3>Connection Settings</h3>
      <div className="setting-item">
        <label>Server URL:</label>
        <input
          type="text"
          value={serverUrl}
          onChange={(e) => onServerUrlChange(e.target.value)}
        />
        <button onClick={onConnect} disabled={isConnected}>
          Connect
        </button>
        <button onClick={onDisconnect} disabled={!isConnected}>
          Disconnect
        </button>
      </div>
    </div>
  )
}
