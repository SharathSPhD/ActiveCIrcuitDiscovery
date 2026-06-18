import { defineConfig } from 'vitest/config';
import preact from '@preact/preset-vite';

export default defineConfig({
  // @ts-expect-error preact preset is Vite-plugin compatible
  plugins: [preact()],
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/unit/**/*.test.{ts,tsx}'],
    setupFiles: ['tests/unit/setup.ts'],
  },
  resolve: {
    alias: {
      '@components': new URL('./src/components/', import.meta.url).pathname,
      '@data': new URL('./src/data/', import.meta.url).pathname,
      '@utils': new URL('./src/utils/', import.meta.url).pathname,
    },
  },
});
