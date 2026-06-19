// Canonical scene order + nav metadata for the multi-act narrative.
export interface SceneMeta {
  id: string;
  num: number;
  nav: string;
  title: string;
}

export const SCENES: SceneMeta[] = [
  { id: 'hook', num: 1, nav: 'Open', title: 'A Neuroscientist’s Tool to Pick AI’s Brain' },
  { id: 'blackbox', num: 2, nav: 'The Black Box', title: 'We built it. We can’t read it.' },
  { id: 'microscope', num: 3, nav: 'The Microscope', title: 'A microscope for artificial minds' },
  { id: 'circuit', num: 4, nav: 'Circuits', title: 'Features, circuits, and the name-mover' },
  { id: 'testing', num: 5, nav: 'Poking It', title: 'How you prove a part matters' },
  { id: 'predict', num: 6, nav: 'Prediction Machine', title: 'The brain that predicts' },
  { id: 'efe', num: 7, nav: 'Explore vs Exploit', title: 'One equation for curiosity' },
  { id: 'method', num: 8, nav: 'The Agent', title: 'The meeting point: an agent that experiments' },
  { id: 'explore', num: 9, nav: 'Watch It Think', title: 'Watch the agent explore' },
  { id: 'results', num: 10, nav: 'Does It Work?', title: 'Does it find the circuit?' },
  { id: 'honesty', num: 11, nav: 'The Honest Catch', title: 'The 1255% that isn’t what it looks like' },
  { id: 'limits', num: 12, nav: 'Where It Breaks', title: 'Where the method breaks' },
  { id: 'domains', num: 13, nav: 'Layers', title: 'Different thoughts, different layers' },
  { id: 'implications', num: 14, nav: 'Why It Matters', title: 'Why this matters' },
  { id: 'reading', num: 15, nav: 'Go Deeper', title: 'Read more' },
];
