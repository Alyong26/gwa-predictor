import PwaRegister from './pwa-register'

export const metadata = {
  title: 'MyGWAP',
  description: 'General Weighted Average predictor for academic planning.',
  manifest: '/manifest.json',
  appleWebApp: {
    title: 'MyGWAP',
  },
  icons: {
    icon: '/icon-192.png',
    apple: '/icon-192.png',
  }
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{margin:0,fontFamily:'Arial, sans-serif'}}>
        <PwaRegister />
        {children}
      </body>
    </html>
  )
}
