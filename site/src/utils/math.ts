// Small numeric helpers used by charts + unit-tested in isolation.

/** Shannon entropy (nats) of a discrete distribution. Normalizes input. */
export function entropy(dist: number[]): number {
  const sum = dist.reduce((a, b) => a + b, 0);
  if (sum <= 0) return 0;
  let h = 0;
  for (const x of dist) {
    const p = x / sum;
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

/** Cumulative sum array (running total). */
export function cumsum(xs: number[]): number[] {
  const out: number[] = [];
  let acc = 0;
  for (const x of xs) {
    acc += x;
    out.push(acc);
  }
  return out;
}

/** Clamp helper. */
export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Format a small KL value for display. */
export function fmtKL(x: number): string {
  if (x === 0) return '0';
  if (x < 1e-3) return x.toExponential(1);
  return x.toFixed(4);
}

/** Format a percentage (efficiency). */
export function fmtPct(x: number, digits = 1): string {
  return `${x.toFixed(digits)}%`;
}
