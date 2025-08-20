export const translations = {
  en: {
    'OpenFace-3.0': 'OpenFace-3.0',
    'Overview': 'Overview',
    'Internals': 'Internals',
    'Real-time Facial Analysis': 'Real-time Facial Analysis',
    'Emotion detection and gaze tracking with OpenFace-3.0': 'Emotion detection and gaze tracking with OpenFace-3.0',
    'OpenFace Pipeline Internals': 'OpenFace Pipeline Internals',
    'Understanding the data flow through the facial analysis pipeline': 'Understanding the data flow through the facial analysis pipeline',
    'Play Animation': 'Play Animation',
    'Pause Animation': 'Pause Animation',
    'Reset': 'Reset',
    'Connection Settings': 'Connection Settings',
    'Server URL:': 'Server URL:',
    'Connect': 'Connect',
    'Disconnect': 'Disconnect'
  },
  no: {
    'OpenFace-3.0': 'OpenFace-3.0',
    'Overview': 'Oversikt',
    'Internals': 'Interndetaljer',
    'Real-time Facial Analysis': 'Sanntids Ansiktsanalyse',
    'Emotion detection and gaze tracking with OpenFace-3.0': 'Følelsesgjenkjenning og blikksporing med OpenFace-3.0',
    'OpenFace Pipeline Internals': 'OpenFace Pipeline Interndetaljer',
    'Understanding the data flow through the facial analysis pipeline': 'Forstå datastrømmen gjennom ansiktsanalysepipeline',
    'Play Animation': 'Spill Animasjon',
    'Pause Animation': 'Pause Animasjon',
    'Reset': 'Tilbakestill',
    'Connection Settings': 'Tilkoblingsinnstillinger',
    'Server URL:': 'Server URL:',
    'Connect': 'Koble til',
    'Disconnect': 'Koble fra'
  }
}

export function useTranslation(language: string = 'en') {
  const t = (key: string): string => {
    const lang = language as keyof typeof translations
    const langTranslations = translations[lang]
    if (!langTranslations) return key
    return (langTranslations as any)[key] || key
  }
  
  return { t }
}
