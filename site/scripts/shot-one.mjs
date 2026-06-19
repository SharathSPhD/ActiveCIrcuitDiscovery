import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
const path = process.argv[2] ?? '';
const name = process.argv[3] ?? 'shot';
const BASE = 'http://localhost:4321/ActiveCIrcuitDiscovery/';
mkdirSync('shots', { recursive: true });
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1100, height: 1200 }, deviceScaleFactor: 2 });
await p.goto(BASE + path, { waitUntil: 'networkidle' });
// scroll through to trigger all IntersectionObserver reveals
const h = await p.evaluate(() => document.body.scrollHeight);
for (let y = 0; y < h; y += 600) { await p.evaluate((yy) => window.scrollTo(0, yy), y); await p.waitForTimeout(120); }
await p.evaluate(() => window.scrollTo(0, 0));
await p.waitForTimeout(800);
await p.screenshot({ path: `shots/${name}.png`, fullPage: true });
await b.close();
console.log('shot', name);
