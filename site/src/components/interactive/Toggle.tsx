import type { JSX } from 'preact';

interface Option<T extends string> {
  value: T;
  label: string;
}
interface ToggleProps<T extends string> {
  options: Option<T>[];
  value: T;
  onChange: (v: T) => void;
  ariaLabel: string;
  testid?: string;
}

/** Accessible segmented control (radio semantics). */
export default function Toggle<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  testid,
}: ToggleProps<T>): JSX.Element {
  return (
    <div class="toggle" role="radiogroup" aria-label={ariaLabel} data-testid={testid}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            type="button"
            role="radio"
            aria-checked={active}
            class={active ? 'toggle-btn active' : 'toggle-btn'}
            data-testid={testid ? `${testid}-${o.value}` : undefined}
            data-value={o.value}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        );
      })}
      <style>{`
        .toggle {
          display: inline-flex;
          gap: 2px;
          padding: 3px;
          border: 1px solid var(--hairline);
          border-radius: 999px;
          background: var(--bg-inset);
          flex-wrap: wrap;
        }
        .toggle-btn {
          appearance: none;
          border: none;
          background: transparent;
          color: var(--ink-faint);
          font: inherit;
          font-size: 0.85rem;
          padding: 0.35rem 0.9rem;
          border-radius: 999px;
          cursor: pointer;
          transition: background 0.15s ease, color 0.15s ease;
        }
        .toggle-btn:hover { color: var(--ink-soft); }
        .toggle-btn.active {
          background: linear-gradient(90deg, rgba(34,211,238,0.18), rgba(168,85,247,0.18));
          color: var(--ink);
          box-shadow: inset 0 0 0 1px rgba(34,211,238,0.35);
        }
      `}</style>
    </div>
  );
}
