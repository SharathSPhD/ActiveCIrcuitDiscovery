'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const PARTS = [
  { href: '/mech-interp', label: 'I · Mech Interp' },
  { href: '/active-inference', label: 'II · The Bridge' },
  { href: '/results', label: 'III · Results' },
  { href: '/demo', label: 'IV · Live Demo' },
  { href: '/qa', label: 'V · Q&A' },
];

export default function TopNav() {
  const path = usePathname();
  return (
    <nav className="topnav" aria-label="Site">
      <Link href="/" className="brand">
        <span className="dot" aria-hidden />
        <span>Active Circuit Discovery · The Talk</span>
      </Link>
      <div className="navlinks">
        {PARTS.map((p) => (
          <Link key={p.href} href={p.href} className={path?.startsWith(p.href) ? 'active' : ''}>
            {p.label}
          </Link>
        ))}
        <a className="pill" href="https://www.mdpi.com/2073-8994/18/6/1043" target="_blank" rel="noopener">
          Paper
        </a>
      </div>
    </nav>
  );
}
