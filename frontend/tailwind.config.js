/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: { accent: '#2563eb', accent2: '#7c3aed', ink: '#0f172a' },
    },
  },
  plugins: [],
}
