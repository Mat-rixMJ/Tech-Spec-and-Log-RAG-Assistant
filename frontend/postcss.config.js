import { fileURLToPath } from 'node:url'

// Explicit config path so Tailwind finds custom tokens regardless of cwd.
export default {
  plugins: {
    tailwindcss: { config: fileURLToPath(new URL('./tailwind.config.js', import.meta.url)) },
    autoprefixer: {},
  },
}
