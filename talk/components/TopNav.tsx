'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const PARTS = [
  { href: '/', label: '0 · Opening' },
  { href: '/mech-interp', label: 'I · Black Box' },
  { href: '/active-inference', label: 'II · The Bridge' },
  { href: '/results', label: 'III · Evidence' },
  { href: '/demo', label: 'IV · Live Lab' },
  { href: '/qa', label: 'V · Questions' },
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
          <Link
            key={p.href}
            href={p.href}
            className={(p.href === '/' ? path === '/' : path?.startsWith(p.href)) ? 'active' : ''}
          >
            {p.label}
          </Link>
        ))}
        <a className="pill" href="https://www.mdpi.com/2073-8994/18/6/1043" target="_blank" rel="noopener">
          Paper
        </a>
        <Link className="pill" href="/author">
          Author
        </Link>
      </div>
    </nav>
  );
}
