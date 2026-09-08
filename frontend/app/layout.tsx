import type { ReactNode } from "react";
import type { Metadata } from "next";
import localFont from "next/font/local";

import "./globals.css";

const manrope = localFont({ src: "../public/fonts/manrope-variable.ttf", variable: "--font-manrope", display: "swap", weight: "200 800" });
export const metadata: Metadata = {
  title: { default: "Friday — Get your devices working again", template: "%s · Friday" },
  description: "Describe the problem. Friday finds relevant manufacturer manual evidence and guides you through troubleshooting by text or voice.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={manrope.variable}>
      <body>
        {children}
      </body>
    </html>
  );
}
