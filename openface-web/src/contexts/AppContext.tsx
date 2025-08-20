'use client'

import { createContext, useContext, useState, ReactNode } from 'react'

export type TabType = 'overview' | 'internals'

interface AppContextType {
  currentTab: TabType
  setCurrentTab: (tab: TabType) => void
  currentLanguage: string
  setCurrentLanguage: (language: string) => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [currentTab, setCurrentTab] = useState<TabType>('overview')
  const [currentLanguage, setCurrentLanguage] = useState('en')

  return (
    <AppContext.Provider value={{
      currentTab,
      setCurrentTab,
      currentLanguage,
      setCurrentLanguage
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider')
  }
  return context
}
