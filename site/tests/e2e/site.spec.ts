import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { SCENES } from '../../src/utils/scenes';

const dataDir = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'src', 'data');
const timelineGemma = JSON.parse(readFileSync(resolve(dataDir, 'timeline_gemma.json'), 'utf8'));
const stats = JSON.parse(readFileSync(resolve(dataDir, 'stats.json'), 'utf8'));

test.describe('structure', () => {
  test('all 11 scene anchors render in order', async ({ page }) => {
    await page.goto('./');
    for (const s of SCENES) {
      await expect(page.locator(`section#${s.id}`)).toHaveCount(1);
    }
  });

  test('hero shows the particle brain (or its fallback)', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByTestId('particle-brain')).toBeVisible();
  });
});

test.describe('Scene 5 — agent timeline', () => {
  test('default readout matches step 0 entropy from the data', async ({ page }) => {
    await page.goto('./#explore');
    const tl = page.getByTestId('agent-timeline');
    await tl.scrollIntoViewIfNeeded();
    const expected = timelineGemma.per_prompt[0].agent_entropy_history[0].toFixed(4);
    await expect(page.getByTestId('ro-entropy')).toHaveText(expected);
    await expect(page.getByTestId('ro-action')).toHaveText(/Ablation/);
  });

  test('clicking a step bar updates the entropy readout to that step', async ({ page }) => {
    await page.goto('./#explore');
    await page.getByTestId('agent-timeline').scrollIntoViewIfNeeded();
    const expected = timelineGemma.per_prompt[0].agent_entropy_history[5].toFixed(4);
    await expect(async () => {
      await page.getByTestId('tl-bar-5').click();
      await expect(page.getByTestId('ro-step')).toContainText('6'); // step index 5 -> "6 / 20"
    }).toPass();
    await expect(page.getByTestId('ro-entropy')).toHaveText(expected);
  });

  test('switching model re-renders the trajectory', async ({ page }) => {
    await page.goto('./#explore');
    await page.getByTestId('agent-timeline').scrollIntoViewIfNeeded();
    const llama = page.getByTestId('tl-model-llama');
    await expect(async () => {
      await llama.click();
      await expect(llama).toHaveAttribute('aria-checked', 'true');
    }).toPass();
    await expect(page.getByTestId('ro-entropy')).toBeVisible();
  });
});

test.describe('Scene 6 — efficiency', () => {
  test('POMDP beats random on Gemma IOI (bar width)', async ({ page }) => {
    await page.goto('./#results');
    await page.getByTestId('efficiency-explorer').scrollIntoViewIfNeeded();
    const ai = await page.getByTestId('eff-bar-ai_abl').getAttribute('width');
    const random = await page.getByTestId('eff-bar-random').getAttribute('width');
    expect(parseFloat(ai!)).toBeGreaterThan(parseFloat(random!));
  });

  test('efficiency value matches paper_stats', async ({ page }) => {
    await page.goto('./#results');
    await page.getByTestId('efficiency-explorer').scrollIntoViewIfNeeded();
    const eff = await page.getByTestId('eff-row-ai_abl').getAttribute('data-eff');
    const expected = (stats as any).gemma.ioi.methods.ai_abl.oracle_eff.point;
    expect(Math.abs(parseFloat(eff!) - expected)).toBeLessThan(0.1);
  });

  test('switching to Llama lowers the POMDP efficiency', async ({ page }) => {
    await page.goto('./#results');
    await page.getByTestId('efficiency-explorer').scrollIntoViewIfNeeded();
    const llama = page.getByTestId('eff-model-llama');
    await expect(async () => {
      await llama.click();
      await expect(llama).toHaveAttribute('aria-checked', 'true');
    }).toPass();
    const eff = await page.getByTestId('eff-row-ai_abl').getAttribute('data-eff');
    expect(parseFloat(eff!)).toBeLessThan(60); // 52.1%
  });
});

test.describe('Scene 7 — honesty', () => {
  test('RCK shows random+steering also exceeding 100%', async ({ page }) => {
    await page.goto('./#honesty');
    await page.getByTestId('rck-panel').scrollIntoViewIfNeeded();
    const rs = await page.getByTestId('rck-row-random_steer').getAttribute('data-rck');
    expect(parseFloat(rs!)).toBeGreaterThan(100);
  });

  test('H2a accepted, H2b not supported', async ({ page }) => {
    await page.goto('./#honesty');
    await page.getByTestId('steering-panel').scrollIntoViewIfNeeded();
    await expect(page.getByTestId('h2a-verdict')).toContainText('Accepted');
    await expect(page.getByTestId('h2b-verdict')).toContainText('Not supported');
  });
});

test.describe('accessibility', () => {
  test('skip link is the first focusable element', async ({ page }) => {
    await page.goto('./');
    await page.keyboard.press('Tab');
    await expect(page.locator('a.skip-link')).toBeFocused();
  });
});

test.describe('accessibility — reduced motion', () => {
  test.use({ reducedMotion: 'reduce' });
  test('content still renders with reduced motion', async ({ page }) => {
    await page.goto('./');
    await expect(page.locator('h1')).toContainText('Neuroscientist');
    await expect(page.getByTestId('particle-brain')).toBeVisible();
  });
});
