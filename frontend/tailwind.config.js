/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: '#0b0f19',
        darkCard: '#111827',
        accentBlue: '#3b82f6',
        accentPurple: '#8b5cf6',
      }
    },
  },
  plugins: [],
}
