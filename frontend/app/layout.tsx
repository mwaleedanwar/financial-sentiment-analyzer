import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial Sentiment Demo",
  description: "Live demo UI for a DistilBERT financial sentiment classifier.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
