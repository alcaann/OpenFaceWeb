'use client'

import { I18nProvider } from '@lingui/react'
import { i18n } from '@lingui/core'
import { ReactNode, useEffect, useState } from 'react'

// Import message catalogs
import { messages as enMessages } from '@/locales/en/messages.js'
import { messages as noMessages } from '@/locales/no/messages.js'

interface LinguiProviderProps {
  children: ReactNode
}

export function LinguiProvider({ children }: LinguiProviderProps) {
  console.log('🌐 LinguiProvider rendering...')
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    console.log('🔄 Loading LinguiJS messages...')
    try {
      // Load all message catalogs
      i18n.load({
        en: enMessages,
        no: noMessages,
      })
      console.log('📦 Messages loaded successfully')
      
      // Activate default locale
      i18n.activate('en')
      console.log('✅ LinguiJS activated with locale: en')
      setIsLoaded(true)
    } catch (error) {
      console.error('❌ Error loading LinguiJS:', error)
      setIsLoaded(true) // Still set to loaded to prevent infinite loading
    }
  }, [])

  if (!isLoaded) {
    console.log('⏳ LinguiProvider waiting for messages to load...')
    return <div>{children}</div>
  }

  console.log('🎉 LinguiProvider ready, wrapping with I18nProvider')
  return (
    <I18nProvider i18n={i18n}>
      {children}
    </I18nProvider>
  )
}
