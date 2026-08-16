import type { ReactNode } from "react";

import "./globals.css";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/*
          THESIS: Friday is a conversation first, with a live diagnostic map that turns uncertainty into one safe next action.
          OWN-WORLD: A warm field-console surface of paper, ink navy, signal orange, and amber instrument states; the rail is a measuring tool, not decoration.
          STORY: The owner describes a messy symptom, sees the device and evidence become legible, then reports the result of one cited test.
          FIRST VIEWPORT: A dominant transcript and composer on the left; a persistent device, signal, observation, test, and evidence rail on the right.
          FORM: Code-first diagnostic console, chosen for interaction fidelity; seed key 63548530.
          FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
        */}
        {children}
      </body>
    </html>
  );
}
