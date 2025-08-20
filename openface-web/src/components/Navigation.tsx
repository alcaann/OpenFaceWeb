'use client'

import { useApp, TabType } from '@/contexts/AppContext'
import { ThemeToggle } from '@/components/ThemeToggle'
import { LanguageSelector } from '@/components/LanguageSelector'
import { Eye, Settings } from 'lucide-react'

export function Navigation() {
  const { currentTab, setCurrentTab, currentLanguage, setCurrentLanguage } = useApp()

  const tabs = [
    { id: 'overview' as TabType, name: 'Overview', icon: Eye },
    { id: 'internals' as TabType, name: 'Internals', icon: Settings }
  ]

  return (
    <nav className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Title */}
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                🎥 OpenFace-3.0
              </h1>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex items-center space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setCurrentTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    currentTab === tab.id
                      ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.name}
                </button>
              )
            })}
          </div>

          {/* Controls */}
          <div className="flex items-center space-x-2">
            <LanguageSelector
              currentLanguage={currentLanguage}
              onLanguageChange={setCurrentLanguage}
            />
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  )
}
