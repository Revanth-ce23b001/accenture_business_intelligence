import type { Config } from "tailwindcss";

/**
 * The palette is CLAUDE.md §Stack, not a designer's preference, and each
 * colour carries a meaning the UI must not use it outside of:
 *
 *   accent   #A100FF  Accenture purple — stage chrome, and nothing else
 *   real     #FF6B00  orange — the REAL residual, and anything requiring action
 *   expected grey     the part of a movement the calendar accounts for
 *   verified #00B8A9  teal — verified, structured evidence
 *   unknown  amber    unverifiable: alive, and nobody can check it
 *
 * Using `real` for decoration, or `verified` on a figure nobody verified,
 * would make the colours mean nothing — and the colours are how a reader
 * sees the argument before they read a word of it.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: { DEFAULT: "#A100FF", soft: "#F3E5FF", ink: "#5B0091" },
        real: { DEFAULT: "#FF6B00", soft: "#FFF0E4", ink: "#8A3A00" },
        verified: { DEFAULT: "#00B8A9", soft: "#E1F7F5", ink: "#00594F" },
        unknown: { DEFAULT: "#C77700", soft: "#FFF6E3", ink: "#7A4A00" },
        expected: { DEFAULT: "#8A8F98", soft: "#F1F2F4", ink: "#40454D" },
        ink: { DEFAULT: "#14161A", muted: "#5C636E", faint: "#8A8F98" },
        paper: { DEFAULT: "#FFFFFF", sunk: "#F7F8FA", edge: "#E4E7EC" },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      keyframes: {
        pulseRail: {
          "0%,100%": { opacity: "1" },
          "50%": { opacity: "0.45" },
        },
      },
      animation: { pulseRail: "pulseRail 1.2s ease-in-out infinite" },
    },
  },
  plugins: [],
};
export default config;
