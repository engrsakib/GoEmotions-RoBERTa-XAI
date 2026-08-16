import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'GoEmotions RoBERTa XAI',
  description:
    'Fine-tuned sentiment classifier with token-level attribution heatmaps',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
