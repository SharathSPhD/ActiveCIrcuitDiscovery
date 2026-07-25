'use client';

/**
 * Deck — full-screen slide framework for the talk.
 *
 * <Deck part="Part I" title="Mech Interp">
 *   <Slide steps={2} notes={<>long-form content</>}>
 *     <h1>Big claim</h1>
 *     <Reveal at={1}>appears on first advance</Reveal>
 *     <Reveal at={2}>appears on second advance</Reveal>
 *   </Slide>
 * </Deck>
 *
 * Keys: → / ↓ / Space / PageDown advance (through reveals, then slides);
 *       ← / ↑ / PageUp go back; Home/End jump; N toggles the notes panel.
 * URL hash (#3) keeps the position across reloads.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

interface SlideState {
  step: number; // current reveal step for this slide
  active: boolean;
  visited: boolean;
}

const SlideCtx = createContext<SlideState>({ step: 99, active: true, visited: true });

export function Reveal({
  at,
  children,
  className = '',
  style,
}: {
  at: number;
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  const { step } = useContext(SlideCtx);
  return (
    <div className={`rv ${step >= at ? 'rv-on' : ''} ${className}`} style={style}>
      {children}
    </div>
  );
}

export interface SlideProps {
  children: React.ReactNode;
  /** number of build steps (max `at` used by Reveals inside) */
  steps?: number;
  /** long-form notes shown in the side panel (press N) */
  notes?: React.ReactNode;
  /** don't mount children until the slide is first visited (heavy embeds) */
  lazy?: boolean;
  /** extra class on the slide, e.g. 'sl-center' */
  className?: string;
}

export function Slide(_props: SlideProps) {
  // Rendered by Deck; never directly.
  return null;
}

export default function Deck({
  part,
  title,
  children,
}: {
  part: string;
  title: string;
  children: React.ReactNode;
}) {
  const slides = useMemo(
    () =>
      React.Children.toArray(children).filter(
        (c): c is React.ReactElement<SlideProps> => React.isValidElement(c)
      ),
    [children]
  );
  const n = slides.length;

  const [pos, setPos] = useState<{ s: number; step: number }>({ s: 0, step: 0 });
  const [visited, setVisited] = useState<Set<number>>(() => new Set([0]));
  const [notesOpen, setNotesOpen] = useState(false);
  const posRef = useRef(pos);
  posRef.current = pos;

  const stepsOf = useCallback((i: number) => slides[i]?.props.steps ?? 0, [slides]);

  // restore from hash (on mount and on manual hash edits)
  useEffect(() => {
    const apply = () => {
      const m = /^#(\d+)$/.exec(window.location.hash);
      if (!m) return;
      const s = Math.min(Math.max(parseInt(m[1], 10) - 1, 0), n - 1);
      if (s === posRef.current.s) return;
      setPos({ s, step: stepsOf(s) });
      setVisited((v) => new Set(v).add(s));
    };
    apply();
    window.addEventListener('hashchange', apply);
    return () => window.removeEventListener('hashchange', apply);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n, stepsOf]);

  useEffect(() => {
    history.replaceState(null, '', `#${pos.s + 1}`);
  }, [pos.s]);

  const go = useCallback(
    (dir: 1 | -1) => {
      setNotesOpen(false);
      setPos((p) => {
        if (dir === 1) {
          if (p.step < stepsOf(p.s)) return { s: p.s, step: p.step + 1 };
          if (p.s < n - 1) {
            const s = p.s + 1;
            setVisited((v) => new Set(v).add(s));
            return { s, step: 0 };
          }
          return p;
        } else {
          if (p.step > 0) return { s: p.s, step: p.step - 1 };
          if (p.s > 0) {
            const s = p.s - 1;
            setVisited((v) => new Set(v).add(s));
            return { s, step: stepsOf(s) };
          }
          return p;
        }
      });
    },
    [n, stepsOf]
  );

  const jump = useCallback(
    (s: number, fullStep = false) => {
      setNotesOpen(false);
      const t = Math.min(Math.max(s, 0), n - 1);
      setVisited((v) => new Set(v).add(t));
      setPos({ s: t, step: fullStep ? stepsOf(t) : 0 });
    },
    [n, stepsOf]
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowDown':
        case ' ':
        case 'PageDown':
          e.preventDefault();
          go(1);
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
        case 'PageUp':
          e.preventDefault();
          go(-1);
          break;
        case 'Home':
          e.preventDefault();
          jump(0);
          break;
        case 'End':
          e.preventDefault();
          jump(n - 1, true);
          break;
        case 'n':
        case 'N':
          e.preventDefault();
          setNotesOpen((o) => !o);
          break;
        case 'Escape':
          setNotesOpen(false);
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go, jump, n]);

  const cur = slides[pos.s];
  const hasNotes = !!cur?.props.notes;

  return (
    <div className="deck" role="region" aria-roledescription="slide deck" aria-label={`${part} — ${title}`}>
      {slides.map((sl, i) => {
        const active = i === pos.s;
        const seen = visited.has(i);
        const state: SlideState = {
          step: active ? pos.step : stepsOf(i),
          active,
          visited: seen,
        };
        const mount = !sl.props.lazy || seen;
        return (
          <SlideCtx.Provider key={i} value={state}>
            <section
              className={`slide ${sl.props.className ?? ''} ${active ? 'slide-on' : ''}`}
              aria-hidden={!active}
            >
              <div className="slide-body">{mount ? sl.props.children : null}</div>
            </section>
          </SlideCtx.Provider>
        );
      })}

      {/* HUD */}
      <div className="deck-hud">
        <span className="hud-part">{part}</span>
        <div className="hud-dots" aria-hidden>
          {slides.map((_, i) => (
            <button
              key={i}
              className={`dot ${i === pos.s ? 'on' : ''}`}
              onClick={() => jump(i)}
              tabIndex={-1}
              aria-label={`Slide ${i + 1}`}
            />
          ))}
        </div>
        <div className="hud-right">
          {hasNotes && (
            <button className={`hud-btn ${notesOpen ? 'on' : ''}`} onClick={() => setNotesOpen((o) => !o)}>
              Notes · N
            </button>
          )}
          <button className="hud-btn" onClick={() => go(-1)} aria-label="Previous">
            ←
          </button>
          <span className="hud-count">
            {pos.s + 1} / {n}
          </span>
          <button className="hud-btn" onClick={() => go(1)} aria-label="Next">
            →
          </button>
        </div>
      </div>

      {/* Notes drawer */}
      {hasNotes && (
        <>
          <div className={`notes-veil ${notesOpen ? 'open' : ''}`} onClick={() => setNotesOpen(false)} />
          <aside className={`notes-panel ${notesOpen ? 'open' : ''}`} aria-label="Slide notes">
            <div className="notes-head">
              <span>
                Notes — slide {pos.s + 1} / {n}
              </span>
              <button className="hud-btn" onClick={() => setNotesOpen(false)}>
                Close · Esc
              </button>
            </div>
            <div className="notes-body">{cur?.props.notes}</div>
          </aside>
        </>
      )}
    </div>
  );
}
