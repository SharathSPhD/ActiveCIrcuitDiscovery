// Validates the safety net: with no DGX reachable the page must show OFFLINE
// and the SAME Run button must drive a replay of the paper's recorded runs.
import { chromium } from 'playwright';

const BASE = process.env.BASE ?? 'http://localhost:3100';
const fail = [];
const ok = (m) => console.log('  PASS  ' + m);
const bad = (m) => { console.log('  FAIL  ' + m); fail.push(m); };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));

await page.goto(BASE + '/demo', { waitUntil: 'domcontentloaded', timeout: 90000 });

console.log('\n[1] badge must report OFFLINE when DGX_TUNNEL_URL is unset');
await page.waitForFunction(
  () => /LIVE ·|OFFLINE/.test(document.body.innerText),
  { timeout: 60000 },
).catch(() => {});
const badge = await page.evaluate(() => {
  const m = document.body.innerText.match(/(LIVE · [^\n]*|OFFLINE[^\n]*|PROBING DGX[^\n]*)/);
  return m ? m[1].trim() : '(none)';
});
console.log('        badge: ' + badge);
/^OFFLINE/.test(badge) ? ok('degrades to OFFLINE · REPLAY MODE') : bad('expected OFFLINE, got: ' + badge);

console.log('\n[2] the same button must still run a replay');
const btn = page.locator('button').first();
console.log('        clicking: "' + (await btn.innerText()).trim() + '"');
await btn.click();
try {
  await page.waitForFunction(() => /\bstep\s*\d/i.test(document.body.innerText), { timeout: 60000 });
  ok('replay steps render');
} catch { bad('replay produced no steps'); }

await page.waitForTimeout(10000);
const n = await page.evaluate(() => {
  const r = [...document.body.innerText.matchAll(/\bstep\s*(\d+)/gi)].map((m) => +m[1]);
  return r.length ? Math.max(...r) : 0;
});
console.log(`        highest replay step: ${n}`);
n >= 3 ? ok(`replay advanced to step ${n}`) : bad(`replay stalled at ${n}`);

const phase = await page.evaluate(() => {
  const m = document.body.innerText.match(/(replaying a real recorded run[^\n]*|graph ready[^\n]*)/);
  return m ? m[1] : '(no phase line)';
});
console.log('        phase: ' + phase);
/replaying a real recorded run/.test(phase) ? ok('replay sourced from recorded runs') : bad('unexpected phase: ' + phase);

await page.screenshot({ path: '/tmp/acd/shot-demo-offline.png' });
errs.length === 0 ? ok('no page errors') : bad('page errors: ' + errs.slice(0, 5).join(' | '));

await browser.close();
console.log('\n' + (fail.length ? `RESULT: ${fail.length} FAILURE(S)` : 'RESULT: FALLBACK OK'));
process.exit(fail.length ? 1 : 0);
