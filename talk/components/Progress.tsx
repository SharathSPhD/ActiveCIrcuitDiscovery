'use client';

import { useEffect, useRef } from 'react';

export default function Progress() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      if (ref.current) ref.current.style.width = `${max > 0 ? (h.scrollTop / max) * 100 : 0}%`;
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);
  return <div className="progress" ref={ref} aria-hidden />;
}
