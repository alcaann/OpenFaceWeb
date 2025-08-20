'use client'

import { Navigation } from '@/components/Navigation'
import { OverviewPage } from '@/components/OverviewPage'
import { InternalsPage } from '@/components/InternalsPage'
import { useApp } from '@/contexts/AppContext'
import { useEffect } from 'react'

export default function Home() {
  console.log('🏠 Home component rendering...')
  
  const { currentTab } = useApp()
  console.log('📍 Current tab:', currentTab)

  useEffect(() => {
    console.log('✅ Home component mounted successfully')
    return () => {
      console.log('🔄 Home component unmounting')
    }
  }, [])

  console.log('🎨 Rendering main content...')

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Navigation />
      <main>
        {currentTab === 'overview' && <OverviewPage />}
        {currentTab === 'internals' && <InternalsPage />}
      </main>
    </div>
  )
}
