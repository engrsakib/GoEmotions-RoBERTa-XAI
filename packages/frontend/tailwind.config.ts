/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        ink: '#0f172a',
        panel: '#111827',
        accent: '#38bdf8',
        positive: '#22c55e',
        negative: '#ef4444',
      },
    },
  },
  plugins: [],
};
