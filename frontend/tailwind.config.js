/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#FAF6ED',
        surface: '#FFF9EA',
        'surface-container': '#F3EDDE',
        'surface-container-high': '#EEE8D9',
        'surface-container-low': '#F9F3E4',
        'surface-container-lowest': '#FFFFFF',
        espresso: '#1E1B18',
        'espresso-surface': '#26201D',
        cognac: '#B45309',
        amberGlow: '#D97706',
        'primary-container': '#F59E0B',
        'on-primary-container': '#613B00',
        sandBorder: '#E6DFD5',
        mutedSand: '#F3EFE6',
        slateInk: '#0F172A',
        outline: '#867461',
        'outline-variant': '#D8C3AD',
        tertiary: '#006C49',
        'tertiary-container': '#30C88F',
        error: '#BA1A1A',
        'error-container': '#FFDAD6',
      },
      borderRadius: {
        'xl': '1.25rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
        'full': '9999px',
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      boxShadow: {
        'warm': '0 12px 32px -8px rgba(184, 169, 142, 0.18), 0 2px 8px 0 rgba(184, 169, 142, 0.08)',
        'warm-sm': '0 2px 8px 0 rgba(184, 169, 142, 0.12)',
        'dark-card': '0 20px 40px -12px rgba(30, 30, 36, 0.28)',
      }
    },
  },
  plugins: [],
}
