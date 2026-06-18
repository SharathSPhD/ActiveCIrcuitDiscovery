// Shared chart/theme constants (kept in sync with global.css tokens).

export const COLORS = {
  bg: '#0a0e14',
  bgPanel: '#0d1420',
  ink: '#e6edf3',
  inkSoft: '#aebacb',
  inkFaint: '#6b7a90',
  hairline: '#1d2839',
  cyan: '#22d3ee',
  violet: '#a855f7',
  amber: '#f59e0b',
  rose: '#fb7185',
  accept: '#34d399',
  reject: '#fb7185',
} as const;

// Intervention action -> color (matches paper's three actions).
export const ACTION_COLORS: Record<string, string> = {
  ablation: '#f59e0b',
  activation_patching: '#22d3ee',
  feature_steering: '#a855f7',
};

export const ACTION_LABELS: Record<string, string> = {
  ablation: 'Ablation',
  activation_patching: 'Patching',
  feature_steering: 'Steering',
};

// Method -> color + display label (used across comparison charts).
export const METHODS = {
  ai: { label: 'POMDP (multi-action)', color: '#22d3ee' },
  ai_abl: { label: 'POMDP (ablation-only)', color: '#34d399' },
  bandit: { label: 'Bandit', color: '#a855f7' },
  ucb: { label: 'UCB', color: '#c084fc' },
  eap: { label: 'EAP (attribution)', color: '#f59e0b' },
  greedy: { label: 'Greedy', color: '#fb7185' },
  random: { label: 'Random', color: '#6b7a90' },
  oracle: { label: 'Oracle', color: '#e6edf3' },
} as const;

export type MethodKey = keyof typeof METHODS;
export type ModelKey = 'gemma' | 'llama';

export const MODEL_LABELS: Record<ModelKey, string> = {
  gemma: 'Gemma-2-2B',
  llama: 'Llama-3.2-1B',
};

export const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
