import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'OpenFace-3.0 Web Client',
  description: 'Real-time facial analysis web application',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
