// Local e2e for the slide-deck rework. Run from talk/: node e2e/deck-local.mjs
// BASE defaults to http://localhost:3100
import { chromium } from 'playwright';

const BASE = process.env.BASE || 'http://localhost:3100';
const DECKS = [
  { path: '/', slides: 5 },
  { path: '/mech-interp', slides: 16 },
  { path: '/active-inference', slides: 14 },
  { path: '/results', slides: 14 },
];

let failures = 0;
const ok = (name, cond, extra = '') => {
  console.log(`${cond ? '  ✓' : '  ✗'} ${name}${extra ? ' — ' + extra : ''}`);
  if (!cond) failures++;
};

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(`console: ${m.text()}`);
});

for (const d of DECKS) {
  console.log(`\n${d.path}`);
  await page.goto(BASE + d.path, { waitUntil: 'networkidle' });
  const deck = page.locator('.deck');
  ok('deck present', (await deck.count()) === 1);
  const nSlides = await page.locator('.slide').count();
  ok(`slide count = ${d.slides}`, nSlides === d.slides, `got ${nSlides}`);
  ok('first slide active', await page.locator('.slide').first().evaluate((el) => el.classList.contains('slide-on')));
  const count = await page.locator('.hud-count').innerText();
  ok(`counter reads 1 / ${d.slides}`, count.trim() === `1 / ${d.slides}`, count.trim());

  // advance through everything with ArrowRight; count distinct slides seen
  let advances = 0;
  const maxAdv = d.slides * 12;
  const activeIndex = async () =>
    page.locator('.slide').evaluateAll((els) => els.findIndex((e) => e.classList.contains('slide-on')));
  let idx = await activeIndex();
  while (idx < d.slides - 1 && advances < maxAdv) {
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(40);
    idx = await activeIndex();
    advances++;
  }
  ok('keyboard reaches last slide', idx === d.slides - 1, `after ${advances} presses`);
  ok('hash tracks position', new URL(page.url()).hash === `#${d.slides}`, page.url());

  // all reveals on the last slide should be on after End
  await page.keyboard.press('End');
  await page.waitForTimeout(200);
  const revealsOff = await page.locator('.slide.slide-on .rv:not(.rv-on)').count();
  ok('all reveals shown at End', revealsOff === 0, `${revealsOff} hidden`);

  // back navigation
  await page.keyboard.press('Home');
  await page.waitForTimeout(120);
  ok('Home returns to slide 1', (await activeIndex()) === 0);

  // notes panel on a slide that has notes (slide 1 everywhere)
  const notesBtn = page.locator('.deck-hud .hud-btn', { hasText: 'Notes' });
  if ((await notesBtn.count()) > 0) {
    await page.keyboard.press('n');
    await page.waitForTimeout(350);
    ok('notes panel opens (N)', await page.locator('.notes-panel.open').isVisible());
    const notesText = (await page.locator('.notes-body').innerText()).trim();
    ok('notes have long-form content', notesText.length > 100, `${notesText.length} chars`);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(350);
    ok('Esc closes notes', (await page.locator('.notes-panel.open').count()) === 0);
  } else {
    ok('notes button present on slide 1', false);
  }

  // chapter contents (TOC)
  await page.keyboard.press('t');
  await page.waitForTimeout(350);
  ok('TOC opens (T)', await page.locator('.toc-panel.open').isVisible());
  const tocCards = await page.locator('.toc-card').count();
  ok(`TOC lists all ${d.slides} slides`, tocCards === d.slides, `got ${tocCards}`);
  const briefs = await page.locator('.toc-card .tc-brief').count();
  ok('TOC cards have briefs', briefs === d.slides, `got ${briefs}`);
  await page.locator('.toc-card').nth(2).click();
  await page.waitForTimeout(350);
  ok('TOC click jumps to slide 3', (await activeIndex()) === 2);
  ok('TOC closed after jump', (await page.locator('.toc-panel.open').count()) === 0);
}

// results deck: charts render (svg inside fig panels)
console.log('\n/results charts');
await page.goto(BASE + '/results#4', { waitUntil: 'networkidle' });
await page.waitForTimeout(600);
ok('EffBars svg renders', (await page.locator('.slide.slide-on svg').count()) > 0);

// mech-interp lazy iframe: not mounted at slide 1, mounted at slide 6
console.log('\n/mech-interp lazy embed');
await page.goto(BASE + '/mech-interp', { waitUntil: 'networkidle' });
ok('iframe NOT mounted before visit', (await page.locator('iframe').count()) === 0);
await page.goto(BASE + '/mech-interp#8', { waitUntil: 'networkidle' });
await page.waitForTimeout(600);
ok('iframe mounted on slide 8', (await page.locator('iframe').count()) === 1);

// demo page: still functional, replay fallback badge appears (no DGX from here)
console.log('\n/demo');
await page.goto(BASE + '/demo', { waitUntil: 'networkidle' });
await page.waitForTimeout(4000);
const badge = await page.locator('.badge-live').innerText().catch(() => '');
ok('demo badge rendered', badge.length > 0, badge.replace(/\s+/g, ' '));
ok('demo app present', (await page.locator('button').count()) > 0);

// steering lab present (offline → recorded dose-response)
ok('steering lab rendered', (await page.locator('text=steering laboratory').count()) > 0);

// qa page
console.log('\n/qa');
await page.goto(BASE + '/qa', { waitUntil: 'networkidle' });
ok('QA cards present', (await page.locator('.qa-card').count()) > 10);

const realErrors = errors.filter(
  (e) => !/net::|Failed to load resource|ERR_|favicon|third-party cookie|neuronpedia|fonts.g/i.test(e)
);
console.log(`\nconsole/page errors (filtered): ${realErrors.length}`);
realErrors.forEach((e) => console.log('   ', e.slice(0, 200)));
if (realErrors.length) failures++;

await browser.close();
console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
