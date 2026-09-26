import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AeroReliability Fleet Health",
  description: "Predictive maintenance fleet health dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
