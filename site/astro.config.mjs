import { defineConfig } from 'astro/config';
import preact from '@astrojs/preact';
import mdx from '@astrojs/mdx';

// Project Pages site: served at https://SharathSPhD.github.io/ActiveCIrcuitDiscovery/
const SITE = 'https://SharathSPhD.github.io';
const BASE = '/ActiveCIrcuitDiscovery';

// Allow overriding base to '/' for local-root or custom-domain builds via env.
const base = process.env.SITE_BASE ?? BASE;

export default defineConfig({
  site: SITE,
  base,
  trailingSlash: 'ignore',
  integrations: [preact({ compat: true }), mdx()],
  vite: {
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            d3: ['d3'],
            three: ['three'],
          },
        },
      },
    },
  },
});
