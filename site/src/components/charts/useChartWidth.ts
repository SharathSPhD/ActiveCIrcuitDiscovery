import { useEffect, useRef, useState } from 'preact/hooks';

/** Track a container's width responsively (SSR-safe). */
export function useChartWidth(fallback = 680) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);

  useEffect(() => {
    if (typeof window === 'undefined' || !ref.current) return;
    const el = ref.current;
    const update = () => setWidth(el.clientWidth || fallback);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [fallback]);

  return { ref, width };
}

export const reduceMotion = (): boolean =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
