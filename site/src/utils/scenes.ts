// Canonical scene order + nav metadata for the single-scroll narrative.
export interface SceneMeta {
  id: string;
  num: number;
  nav: string; // short nav label
  title: string;
}

export const SCENES: SceneMeta[] = [
  { id: 'hook', num: 1, nav: 'The Black Box', title: 'A Neuroscientist’s Tool to Pick AI’s Brain' },
  { id: 'problem', num: 2, nav: 'Too Many Paths', title: 'Why finding a circuit is hard' },
  { id: 'insight', num: 3, nav: 'Free Energy', title: 'A clue from the brain' },
  { id: 'method', num: 4, nav: 'The Agent', title: 'Turning discovery into a decision' },
  { id: 'explore', num: 5, nav: 'Watch It Explore', title: 'Watch the agent think' },
  { id: 'results', num: 6, nav: 'Does It Work?', title: 'Does it actually find the circuit?' },
  { id: 'honesty', num: 7, nav: 'The Honest Catch', title: 'The 1255% that isn’t what it looks like' },
  { id: 'limits', num: 8, nav: 'Where It Breaks', title: 'Where the method breaks' },
  { id: 'domains', num: 9, nav: 'Different Tasks', title: 'Different thoughts live in different layers' },
  { id: 'implications', num: 10, nav: 'Why It Matters', title: 'Why this matters' },
  { id: 'reading', num: 11, nav: 'Go Deeper', title: 'Read more' },
];
