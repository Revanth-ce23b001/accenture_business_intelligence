import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "CaseFile.ai",
  description:
    "An AI business investigator. Dashboards describe; copilots narrate; CaseFile investigates.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">
        <header className="border-b border-paper-edge bg-paper">
          <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-3">
            <Link href="/" className="flex items-baseline gap-3">
              <span className="text-lg font-semibold tracking-tight text-ink">
                CaseFile<span className="text-accent">.ai</span>
              </span>
              <span className="hidden text-xs text-ink-muted sm:inline">
                Dashboards describe. Copilots narrate. CaseFile investigates.
              </span>
            </Link>
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
              Meridian Footwear India · 412 stores
            </span>
          </div>
        </header>
        <main className="mx-auto max-w-[1600px] px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
