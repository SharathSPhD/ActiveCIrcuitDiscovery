// Author + resource links. Fill the `null`s when the URLs are provided.
// Only non-null entries render in the header/footer.

export const AUTHOR = {
  name: 'Dr. Sharath Sathish',
  tagline: 'Active Circuit Discovery',
};

export interface LinkDef {
  key: string;
  label: string;
  href: string | null;
  kind: 'primary' | 'social';
}

export const LINKS: LinkDef[] = [
  // Primary resources
  { key: 'paper', label: 'Paper (PDF)', href: 'https://github.com/SharathSPhD/ActiveCIrcuitDiscovery/blob/main/paper/paper.pdf', kind: 'primary' },
  { key: 'github', label: 'Code & data', href: 'https://github.com/SharathSPhD/ActiveCIrcuitDiscovery', kind: 'primary' },
  { key: 'portfolio', label: 'TechNektar', href: 'https://www.technektar.dev/', kind: 'primary' },
  // Social / scholarly — paste URLs to enable
  { key: 'linkedin', label: 'LinkedIn', href: null, kind: 'social' },
  { key: 'scholar', label: 'Google Scholar', href: null, kind: 'social' },
  { key: 'researchgate', label: 'ResearchGate', href: null, kind: 'social' },
  { key: 'medium', label: 'Medium', href: null, kind: 'social' },
  { key: 'substack', label: 'Substack', href: null, kind: 'social' },
  { key: 'youtube', label: 'YouTube', href: null, kind: 'social' },
];

export const activeLinks = (kind?: 'primary' | 'social') =>
  LINKS.filter((l) => l.href && (!kind || l.kind === kind));
