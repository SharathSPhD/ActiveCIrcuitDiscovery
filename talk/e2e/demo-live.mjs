// End-to-end browser validation of the live demo on the deployed site.
import { chromium } from 'playwright';

const BASE = process.env.BASE ?? 'https://acd-talk.vercel.app';
const fail = [];
const ok = (m) => console.log('  PASS  ' + m);
const bad = (m) => { console.log('  FAIL  ' + m); fail.push(m); };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));

// ---------- every page loads ----------
console.log('\n[1] page loads');
for (const p of ['', '/mech-interp', '/active-inference', '/results', '/demo', '/qa']) {
  const r = await page.goto(BASE + p, { waitUntil: 'domcontentloaded', timeout: 60000 });
  r.status() === 200 ? ok(`${p || '/'} → 200`) : bad(`${p || '/'} → ${r.status()}`);
}

// ---------- demo page: LIVE badge ----------
console.log('\n[2] demo page probes DGX');
await page.goto(BASE + '/demo', { waitUntil: 'domcontentloaded', timeout: 60000 });
try {
  await page.waitForFunction(
    () => /LIVE ·/.test(document.body.innerText) || /OFFLINE/.test(document.body.innerText),
    { timeout: 45000 },
  );
} catch { /* fall through to the assertion below */ }
const badge = await page.evaluate(() => {
  const m = document.body.innerText.match(/(LIVE · [^\n]*|OFFLINE[^\n]*|PROBING DGX[^\n]*)/);
  return m ? m[1].trim() : '(no badge found)';
});
console.log('        badge: ' + badge);
badge.startsWith('LIVE') ? ok('badge shows LIVE + GPU name') : bad('badge is not LIVE: ' + badge);

await page.screenshot({ path: '/tmp/acd/shot-demo-live.png', fullPage: false });

// ---------- run a live episode from the UI ----------
console.log('\n[3] run a live episode from the UI');
const runBtn = page.locator('button', { hasText: /run live|run episode|run$/i }).first();
const btnCount = await page.locator('button').count();
console.log(`        ${btnCount} buttons on page`);
if (await runBtn.count()) {
  const label = (await runBtn.innerText()).trim();
  console.log(`        clicking: "${label}"`);
  const t0 = Date.now();
  await runBtn.click();

  // In LIVE mode rows are labelled with real feature ids (L<layer>_P<pos>_F<idx>);
  // only replay mode labels them "step N". Count either.
  const countRows = () =>
    page.evaluate(() => {
      const t = document.body.innerText;
      const live = new Set(t.match(/\bL\d+_P\d+_F\d+\b/g) ?? []).size;
      const rep = [...t.matchAll(/\bstep\s*(\d+)/gi)].map((m) => +m[1]);
      return { live, replay: rep.length ? Math.max(...rep) : 0 };
    });

  try {
    await page.waitForFunction(
      () => /\bL\d+_P\d+_F\d+\b/.test(document.body.innerText) || /\bstep\s*\d/i.test(document.body.innerText),
      { timeout: 90000 },
    );
    ok(`first step rendered in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  } catch { bad('no step ever rendered in the UI'); }

  const early = await countRows();
  await page.waitForTimeout(15000);
  const late = await countRows();
  const seen = Math.max(late.live, late.replay);
  console.log(`        rows after ~1s: live=${early.live} replay=${early.replay}`);
  console.log(`        rows after ~16s: live=${late.live} replay=${late.replay}`);
  seen >= 3 ? ok(`episode advanced to ${seen} steps`) : bad(`episode stalled at ${seen} steps`);

  // The phase line proves it came from a freshly-built graph, not the replay JSON.
  const phase = await page.evaluate(() => {
    const m = document.body.innerText.match(/(graph ready · [^\n]*|replaying a real recorded run[^\n]*|live run unavailable[^\n]*)/);
    return m ? m[1] : '(no phase line)';
  });
  console.log(`        phase: ${phase}`);
  if (/live run unavailable/.test(phase)) bad('fell back to replay: ' + phase);
  else if (/graph ready/.test(phase)) ok('ran against a live DGX graph');

  await page.screenshot({ path: '/tmp/acd/shot-demo-running.png', fullPage: false });
} else {
  bad('no run button found');
}

// ---------- console health ----------
console.log('\n[4] console errors');
const realErrors = consoleErrors.filter((e) => !/favicon|third-party cookie|Download the React/i.test(e));
realErrors.length === 0 ? ok('no console errors') : bad(`console errors:\n        ` + realErrors.slice(0, 8).join('\n        '));

await browser.close();
console.log('\n' + (fail.length ? `RESULT: ${fail.length} FAILURE(S)` : 'RESULT: ALL CHECKS PASSED'));
process.exit(fail.length ? 1 : 0);
