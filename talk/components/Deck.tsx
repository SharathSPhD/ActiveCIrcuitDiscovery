'use client';

/**
 * Deck — full-screen teaching-slide framework.
 *
 * <Deck part="Chapter I" title="Inside the black box">
 *   <Slide title="Neurons lie" brief="Why single neurons can't be the unit of analysis"
 *          steps={2} notes={<>long-form content</>}>
 *     <h1>Big claim</h1>
 *     <Reveal at={1}>appears on first advance</Reveal>
 *   </Slide>
 * </Deck>
 *
 * Keys: → / ↓ / Space / PageDown advance (through reveals, then slides);
 *       ← / ↑ / PageUp back; Home/End jump; N notes; T chapter contents.
 * URL hash (#3) keeps position across reloads and manual edits.
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
  step: number;
  active: boolean;
  visited: boolean;
}

const SlideCtx = createContext<SlideState>({ step: 99, active: true, visited: true });

/** Read the current slide's reveal step — lets visuals advance stage-by-stage with the clicker. */
export function useSlideStep(): number {
  return useContext(SlideCtx).step;
}

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

/** Standard bottom-of-slide bridge to the next idea. */
export function NextLead({ children }: { children: React.ReactNode }) {
  return (
    <div className="next-lead">
      <span className="nl-arrow" aria-hidden>⟶</span>
      <span>{children}</span>
    </div>
  );
}

export interface SlideProps {
  children: React.ReactNode;
  /** short title shown in the chapter contents */
  title?: string;
  /** one-line brief shown in the chapter contents */
  brief?: string;
  /** number of build steps (max `at` used by Reveals inside) */
  steps?: number;
  /** long-form notes shown in the side panel (press N) */
  notes?: React.ReactNode;
  /** don't mount children until the slide is first visited (heavy embeds) */
  lazy?: boolean;
  className?: string;
}

export function Slide(_props: SlideProps) {
  return null; // rendered by Deck
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
  const [tocOpen, setTocOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
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
      setTocOpen(false);
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
      setTocOpen(false);
      const t = Math.min(Math.max(s, 0), n - 1);
      setVisited((v) => new Set(v).add(t));
      setPos({ s: t, step: fullStep ? stepsOf(t) : 0 });
    },
    [n, stepsOf]
  );

  const toggleFullscreen = useCallback(() => {
    const el = rootRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      el.requestFullscreen().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onFs = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFs);
    return () => document.removeEventListener('fullscreenchange', onFs);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return;
      switch (e.key) {
        case 'ArrowRight':
        case 'Right':
        case 'ArrowDown':
        case ' ':
        case 'PageDown':
          e.preventDefault();
          go(1);
          break;
        case 'ArrowLeft':
        case 'Left':
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
          setTocOpen(false);
          setNotesOpen((o) => !o);
          break;
        case 't':
        case 'T':
          e.preventDefault();
          setNotesOpen(false);
          setTocOpen((o) => !o);
          break;
        case 'f':
        case 'F':
          e.preventDefault();
          toggleFullscreen();
          break;
        case 'Escape':
          setNotesOpen(false);
          setTocOpen(false);
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go, jump, n, toggleFullscreen]);

  // click anywhere on the slide advances — except on interactive elements and panels
  const onDeckClick = useCallback(
    (e: React.MouseEvent) => {
      const el = e.target as HTMLElement;
      if (
        el.closest(
          'a, button, input, select, textarea, iframe, [contenteditable], .deck-hud, .notes-panel, .notes-veil, .toc-panel, .toc-veil, svg [role="slider"]'
        )
      )
        return;
      // ignore clicks that are part of a text selection
      if (window.getSelection()?.toString()) return;
      go(1);
    },
    [go]
  );

  const cur = slides[pos.s];
  const hasNotes = !!cur?.props.notes;

  return (
    <div
      ref={rootRef}
      className={`deck ${fullscreen ? 'deck-fs' : ''}`}
      role="region"
      aria-roledescription="slide deck"
      aria-label={`${part} — ${title}`}
      onClick={onDeckClick}
    >
      {slides.map((sl, i) => {
        const active = i === pos.s;
        const seen = visited.has(i);
        // Inactive slides ahead of the current one sit at step 0 (so arriving on them
        // starts the build cleanly); slides behind sit fully revealed (so going back
        // shows them complete). Rendering future slides fully-revealed caused figures
        // to appear and animate away on entry.
        const state: SlideState = {
          step: active ? pos.step : i < pos.s ? stepsOf(i) : 0,
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
        <button className="hud-btn hud-toc" onClick={() => { setNotesOpen(false); setTocOpen((o) => !o); }}>
          ☰ Contents · T
        </button>
        <span className="hud-part">{part}</span>
        <div className="hud-dots" aria-hidden>
          {slides.map((sl, i) => (
            <button
              key={i}
              className={`dot ${i === pos.s ? 'on' : ''}`}
              onClick={() => jump(i)}
              tabIndex={-1}
              aria-label={sl.props.title ?? `Slide ${i + 1}`}
              title={sl.props.title ?? `Slide ${i + 1}`}
            />
          ))}
        </div>
        <div className="hud-right">
          <button className={`hud-btn ${fullscreen ? 'on' : ''}`} onClick={toggleFullscreen} aria-label="Toggle fullscreen">
            {fullscreen ? '⛶ Exit · F' : '⛶ Full · F'}
          </button>
          {hasNotes && (
            <button className={`hud-btn ${notesOpen ? 'on' : ''}`} onClick={() => { setTocOpen(false); setNotesOpen((o) => !o); }}>
              Notes · N
            </button>
          )}
          <button className="hud-btn" onClick={() => go(-1)} aria-label="Previous">←</button>
          <span className="hud-count">{pos.s + 1} / {n}</span>
          <button className="hud-btn" onClick={() => go(1)} aria-label="Next">→</button>
        </div>
      </div>

      {/* Chapter contents overlay */}
      <div className={`toc-veil ${tocOpen ? 'open' : ''}`} onClick={() => setTocOpen(false)} />
      <aside className={`toc-panel ${tocOpen ? 'open' : ''}`} aria-label="Chapter contents">
        <div className="toc-head">
          <div>
            <div className="toc-part">{part}</div>
            <div className="toc-title">{title}</div>
          </div>
          <button className="hud-btn" onClick={() => setTocOpen(false)}>Close · Esc</button>
        </div>
        <div className="toc-grid">
          {slides.map((sl, i) => (
            <button
              key={i}
              className={`toc-card ${i === pos.s ? 'on' : ''} ${visited.has(i) ? 'seen' : ''}`}
              onClick={() => jump(i)}
            >
              <span className="tc-num">{i + 1}</span>
              <span className="tc-title">{sl.props.title ?? `Slide ${i + 1}`}</span>
              {sl.props.brief && <span className="tc-brief">{sl.props.brief}</span>}
            </button>
          ))}
        </div>
      </aside>

      {/* Notes drawer */}
      {hasNotes && (
        <>
          <div className={`notes-veil ${notesOpen ? 'open' : ''}`} onClick={() => setNotesOpen(false)} />
          <aside className={`notes-panel ${notesOpen ? 'open' : ''}`} aria-label="Slide notes">
            <div className="notes-head">
              <span>Notes — {cur?.props.title ?? `slide ${pos.s + 1}`}</span>
              <button className="hud-btn" onClick={() => setNotesOpen(false)}>Close · Esc</button>
            </div>
            <div className="notes-body">{cur?.props.notes}</div>
          </aside>
        </>
      )}
    </div>
  );
}
