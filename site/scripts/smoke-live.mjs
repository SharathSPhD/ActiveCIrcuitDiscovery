import { chromium } from '@playwright/test';

const URL = process.env.LIVE_URL ?? 'https://sharathsphd.github.io/ActiveCIrcuitDiscovery/';
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

await page.goto(URL, { waitUntil: 'networkidle' });

// 1. Title
const title = await page.title();
if (!/Neuroscientist/.test(title)) throw new Error(`bad title: ${title}`);

// 2. Exercise a chart interaction (efficiency model toggle -> Llama lowers POMDP eff)
await page.goto(URL + '#results', { waitUntil: 'networkidle' });
await page.getByTestId('efficiency-explorer').scrollIntoViewIfNeeded();
const llama = page.getByTestId('eff-model-llama');
for (let i = 0; i < 10; i++) {
  await llama.click().catch(() => {});
  if ((await llama.getAttribute('aria-checked')) === 'true') break;
  await page.waitForTimeout(400);
}
const eff = parseFloat((await page.getByTestId('eff-row-ai_abl').getAttribute('data-eff')) ?? 'NaN');
if (!(eff < 60)) throw new Error(`Llama efficiency unexpected: ${eff}`);

// 3. Timeline scrub interaction
await page.goto(URL + '#explore', { waitUntil: 'networkidle' });
await page.getByTestId('agent-timeline').scrollIntoViewIfNeeded();
await page.getByTestId('tl-bar-5').click({ timeout: 5000 }).catch(() => {});

console.log(`✓ live smoke OK — title="${title}", Llama POMDP eff=${eff}%, console errors=${errors.length}`);
if (errors.length) console.log('  console errors:', errors.slice(0, 5));
await browser.close();
