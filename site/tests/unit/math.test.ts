import { describe, it, expect } from 'vitest';
import { entropy, cumsum, clamp, fmtKL, fmtPct } from '../../src/utils/math';

describe('entropy', () => {
  it('is 0 for a certain distribution', () => {
    expect(entropy([1, 0, 0, 0])).toBe(0);
  });
  it('is maximal (ln N) for a uniform distribution', () => {
    expect(entropy([1, 1, 1, 1])).toBeCloseTo(Math.log(4), 10);
  });
  it('normalizes unnormalized input', () => {
    expect(entropy([2, 2])).toBeCloseTo(Math.log(2), 10);
  });
  it('returns 0 for an all-zero vector', () => {
    expect(entropy([0, 0])).toBe(0);
  });
});

describe('cumsum', () => {
  it('produces a running total', () => {
    expect(cumsum([1, 2, 3])).toEqual([1, 3, 6]);
  });
  it('handles empty input', () => {
    expect(cumsum([])).toEqual([]);
  });
});

describe('clamp', () => {
  it('bounds within range', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(99, 0, 10)).toBe(10);
  });
});

describe('formatters', () => {
  it('formats tiny KL in exponential', () => {
    expect(fmtKL(0.00012)).toMatch(/e-/);
  });
  it('formats a percentage', () => {
    expect(fmtPct(81.9798)).toBe('82.0%');
  });
});
