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
  { key: 'paper', label: 'Paper', href: 'https://www.mdpi.com/2073-8994/18/6/1043', kind: 'primary' },
  { key: 'github', label: 'Code & data', href: 'https://github.com/SharathSPhD/ActiveCIrcuitDiscovery', kind: 'primary' },
  { key: 'portfolio', label: 'TechNektar', href: 'https://www.technektar.dev/', kind: 'primary' },
  // Social / scholarly
  { key: 'linkedin', label: 'LinkedIn', href: 'https://www.linkedin.com/in/sharath-s', kind: 'social' },
  { key: 'scholar', label: 'Google Scholar', href: 'https://scholar.google.com/citations?hl=en&user=dcyu5ucAAAAJ', kind: 'social' },
  { key: 'researchgate', label: 'ResearchGate', href: 'https://www.researchgate.net/profile/Sharath-Sathish/research', kind: 'social' },
  { key: 'medium', label: 'Medium', href: 'https://medium.com/@sharath.ai.colab', kind: 'social' },
  { key: 'substack', label: 'Substack', href: 'https://technektar.substack.com/', kind: 'social' },
  { key: 'youtube', label: 'YouTube', href: 'https://www.youtube.com/@SharathS-PhD', kind: 'social' },
  { key: 'podcast', label: 'Podcast', href: 'https://podcasts.apple.com/us/podcast/oscillatory-odyssey-resonance-rising-part-3/id1796260484?i=1000697224806', kind: 'social' },
];

export const activeLinks = (kind?: 'primary' | 'social') =>
  LINKS.filter((l) => l.href && (!kind || l.kind === kind));
