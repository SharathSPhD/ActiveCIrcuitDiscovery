import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

/**
 * Accessibility gate (M6). Scans the full page with axe-core and the key
 * interactive scenes, failing on any serious/critical WCAG 2.1 A/AA violation.
 */
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];

async function scan(page: any) {
  return new AxeBuilder({ page }).withTags(TAGS).analyze();
}

const serious = (vs: any[]) => vs.filter((v) => v.impact === 'serious' || v.impact === 'critical');

test('home page has no serious/critical a11y violations', async ({ page }) => {
  await page.goto('./');
  await page.waitForTimeout(800);
  const results = await scan(page);
  const bad = serious(results.violations);
  if (bad.length) console.log(JSON.stringify(bad.map((v) => ({ id: v.id, nodes: v.nodes.length })), null, 2));
  expect(bad).toEqual([]);
});

for (const anchor of ['explore', 'results', 'honesty', 'domains']) {
  test(`scene #${anchor} has no serious/critical a11y violations`, async ({ page }) => {
    await page.goto(`./#${anchor}`);
    await page.waitForTimeout(1000);
    const results = await scan(page);
    const bad = serious(results.violations);
    if (bad.length) console.log(anchor, JSON.stringify(bad.map((v) => ({ id: v.id, nodes: v.nodes.length, help: v.help })), null, 2));
    expect(bad).toEqual([]);
  });
}
