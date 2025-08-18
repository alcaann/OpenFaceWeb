'use client';

import React, { useState, useEffect, useRef } from 'react';

interface LogEntry {
  timestamp: number;
  level: string;
  message: string;
  client_id?: string;
  event_type?: string;
  data?: any;
}

interface LogConsoleProps {
  socket?: any;
  maxLogs?: number;
  height?: string;
}

const LogConsoleNew: React.FC<LogConsoleProps> = ({ 
  socket, 
  maxLogs = 100, 
  height = '400px' 
}) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState({
    level: 'ALL',
    eventType: 'ALL',
    clientId: 'ALL'
  });
  
  const consoleRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && consoleRef.current) {
      // Use requestAnimationFrame to ensure DOM is updated before scrolling
      requestAnimationFrame(() => {
        if (consoleRef.current) {
          consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
        }
      });
    }
  }, [logs, autoScroll]);

  // Socket event listeners
  useEffect(() => {
    if (!socket) return;

    const handleConnect = () => {
      setIsConnected(true);
      console.log('LogConsole: Connected to WebSocket');
    };

    const handleDisconnect = () => {
      setIsConnected(false);
      console.log('LogConsole: Disconnected from WebSocket');
    };

    const handleConnected = (data: any) => {
      setIsConnected(true);
      
      // Load initial logs if provided
      if (data.recent_logs && Array.isArray(data.recent_logs)) {
        setLogs(prev => [...prev, ...data.recent_logs]);
      }
    };

    const handleLogEntry = (logEntry: LogEntry) => {
      setLogs(prev => {
        const newLogs = [...prev, logEntry];
        // Keep only the most recent logs
        if (newLogs.length > maxLogs) {
          return newLogs.slice(-maxLogs);
        }
        return newLogs;
      });
    };

    const handleLogsResponse = (response: any) => {
      if (response.success && response.logs) {
        setLogs(response.logs);
      }
    };

    // Register event listeners
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('connected', handleConnected);
    socket.on('log_entry', handleLogEntry);
    socket.on('logs_response', handleLogsResponse);

    return () => {
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('connected', handleConnected);
      socket.off('log_entry', handleLogEntry);
      socket.off('logs_response', handleLogsResponse);
    };
  }, [socket, maxLogs]);

  // Request recent logs
  const requestLogs = (count: number = 50) => {
    if (socket && isConnected) {
      socket.emit('request_logs', { count });
    }
  };

  // Clear logs
  const clearLogs = () => {
    setLogs([]);
  };

  // Get unique values for filters
  const getUniqueValues = (key: keyof LogEntry) => {
    const values = logs
      .map(log => log[key])
      .filter((value, index, self) => value && self.indexOf(value) === index)
      .sort();
    return ['ALL', ...values];
  };

  // Filter logs based on current filters
  const filteredLogs = logs.filter(log => {
    if (filter.level !== 'ALL' && log.level !== filter.level) return false;
    if (filter.eventType !== 'ALL' && log.event_type !== filter.eventType) return false;
    if (filter.clientId !== 'ALL' && log.client_id !== filter.clientId) return false;
    return true;
  });

  // Format timestamp
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  };

  // Get log level styles
  const getLogLevelStyles = (level: string) => {
    switch (level.toLowerCase()) {
      case 'error': 
        return {
          color: '#dc2626',
          backgroundColor: '#fef2f2',
          borderColor: '#dc2626'
        };
      case 'warning': 
        return {
          color: '#d97706',
          backgroundColor: '#fffbeb',
          borderColor: '#d97706'
        };
      case 'info': 
        return {
          color: '#2563eb',
          backgroundColor: '#eff6ff',
          borderColor: '#2563eb'
        };
      case 'debug': 
        return {
          color: '#6b7280',
          backgroundColor: '#f9fafb',
          borderColor: '#6b7280'
        };
      default: 
        return {
          color: '#374151',
          backgroundColor: '#ffffff',
          borderColor: '#d1d5db'
        };
    }
  };

  return (
    <div className="w-full border rounded-lg shadow-sm" style={{ contain: 'layout' }}>
      {/* Header */}
      <div className="p-4 border-b bg-gray-50">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold">
            API Console Logs
            <span 
              className={`ml-2 text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}
            >
              {isConnected ? '● Connected' : '● Disconnected'}
            </span>
          </h3>
          
          <div className="flex gap-2">
            <button
              onClick={() => requestLogs(50)}
              disabled={!isConnected}
              className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
            >
              Refresh
            </button>
            <button
              onClick={clearLogs}
              className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex gap-4 text-sm">
          <div>
            <label className="block text-gray-600 mb-1">Level:</label>
            <select
              value={filter.level}
              onChange={(e) => setFilter(prev => ({ ...prev, level: e.target.value }))}
              className="border rounded px-2 py-1"
            >
              {getUniqueValues('level').map(level => (
                <option key={level} value={level}>{level}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-gray-600 mb-1">Event:</label>
            <select
              value={filter.eventType}
              onChange={(e) => setFilter(prev => ({ ...prev, eventType: e.target.value }))}
              className="border rounded px-2 py-1"
            >
              {getUniqueValues('event_type').map(eventType => (
                <option key={eventType} value={eventType}>{eventType}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-gray-600 mb-1">Client:</label>
            <select
              value={filter.clientId}
              onChange={(e) => setFilter(prev => ({ ...prev, clientId: e.target.value }))}
              className="border rounded px-2 py-1"
            >
              {getUniqueValues('client_id').map(clientId => (
                <option key={clientId} value={clientId}>
                  {clientId === 'ALL' ? 'ALL' : clientId?.substring(0, 8) + '...'}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="mr-1"
              />
              Auto-scroll
            </label>
          </div>
        </div>
      </div>

      {/* Console content */}
      <div
        ref={consoleRef}
        className="font-mono text-sm overflow-y-auto bg-gray-900 text-green-400 relative"
        style={{ 
          height,
          maxHeight: height,
          minHeight: '200px',
          overflowAnchor: 'none',
          contain: 'strict',
          isolation: 'isolate'
        }}
        onScroll={(e) => {
          const element = e.currentTarget;
          const isAtBottom = Math.abs(element.scrollHeight - element.scrollTop - element.clientHeight) < 5;
          setAutoScroll(isAtBottom);
        }}
      >
        {filteredLogs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            No logs to display
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {filteredLogs.map((log, index) => {
              const styles = getLogLevelStyles(log.level);
              return (
                <div
                  key={index}
                  className="border-l-4 pl-3 py-1 rounded-r bg-gray-800/50"
                  style={{
                    borderLeftColor: styles.color
                  }}
                >
                  <div className="flex items-start gap-2 flex-wrap">
                    <span className="text-gray-400 text-xs shrink-0">
                      {formatTimestamp(log.timestamp)}
                    </span>
                    <span 
                      className="font-semibold text-xs shrink-0"
                      style={{ color: styles.color }}
                    >
                      [{log.level}]
                    </span>
                    <span className="flex-1 break-words">
                      {log.message}
                    </span>
                  </div>
                  
                  {/* Additional info */}
                  {(log.client_id || log.event_type) && (
                    <div className="flex gap-4 mt-1 text-xs text-gray-500">
                      {log.client_id && (
                        <span>Client: {log.client_id.substring(0, 8)}...</span>
                      )}
                      {log.event_type && (
                        <span>Event: {log.event_type}</span>
                      )}
                    </div>
                  )}
                  
                  {/* Data object */}
                  {log.data && (
                    <details className="mt-1">
                      <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-300">
                        Show data
                      </summary>
                      <pre className="text-xs bg-gray-700 text-gray-300 p-2 rounded mt-1 overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(log.data, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              );
            })}
            <div ref={bottomRef} className="h-1" />
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="px-3 py-2 bg-gray-50 border-t text-xs text-gray-600 flex justify-between">
        <span>
          Showing {filteredLogs.length} of {logs.length} logs
        </span>
        <span>
          {isConnected ? 'Real-time updates enabled' : 'Disconnected - no updates'}
        </span>
      </div>
    </div>
  );
};

export default LogConsoleNew;
