import { i18n } from '@lingui/core'

export const locales = {
  en: 'English',
  no: 'Norsk (Bokmål)',
}

export const defaultLocale = 'en'

i18n.loadLocaleData({
  en: { plurals: (n: number) => (n === 1 ? 'one' : 'other') },
  no: { plurals: (n: number) => (n === 1 ? 'one' : 'other') },
})

/**
 * We do a dynamic import of just the catalog that we need
 * @param locale any locale string
 */
export async function dynamicActivate(locale: string) {
  const { messages } = await import(`./locales/${locale}/messages`)
  i18n.load(locale, messages)
  i18n.activate(locale)
}
