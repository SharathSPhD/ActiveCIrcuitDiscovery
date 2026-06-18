import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const BASE = 'http://localhost:4321/ActiveCIrcuitDiscovery/';
const OUT = 'shots';
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
await page.goto(BASE, { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);

const scenes = ['hook', 'problem', 'explore', 'results', 'honesty', 'domains'];
for (const s of scenes) {
  await page.goto(BASE + '#' + s, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/${s}.png` });
  console.log('shot', s);
}
await browser.close();
console.log('done');
