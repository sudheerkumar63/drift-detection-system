/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#0f172a',
        secondary: '#111827',
        accent: '#3b82f6',
        highlight: '#38bdf8',
      },
      boxShadow: {
        soft: '0 10px 35px rgba(15, 23, 42, 0.25)',
      },
    },
  },
  plugins: [],
};
