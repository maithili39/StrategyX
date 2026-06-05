/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: '#f8fafc',
        darkCard: '#ffffff',
        accentBlue: '#7c3aed',
        accentPurple: '#db2777',
      }
    },
  },
  plugins: [],
}
