import type { FNode, FEdge, FGroup } from './FlowDiagram';

/** ---- Fig 2 (paper): the Active-Inference perception–action loop ---- */
export const aiLoop: { nodes: FNode[]; edges: FEdge[]; groups: FGroup[]; viewW: number; viewH: number } = {
  viewW: 760,
  viewH: 575,
  groups: [
    { id: 'gm', label: 'Generative model  (the agent’s world-model)', color: '#f59e0b', bbox: { x: 12, y: 8, w: 736, h: 92 } },
  ],
  nodes: [
    { id: 'A', x: 26, y: 36, w: 160, h: 52, label: 'A — likelihood\nP(o | s)', group: 'genmodel' },
    { id: 'B', x: 210, y: 36, w: 160, h: 52, label: 'B — transition\nP(s′ | s, a)', group: 'genmodel' },
    { id: 'C', x: 394, y: 36, w: 160, h: 52, label: 'C — preference\nP(o)', group: 'genmodel' },
    { id: 'D', x: 578, y: 36, w: 156, h: 52, label: 'D — prior\nP(s)', group: 'genmodel' },
    { id: 'infer', x: 200, y: 132, w: 360, h: 54, label: '1 · Variational inference\ninfer beliefs  q(s | o)', group: 'agent', emphasis: true },
    { id: 'efe', x: 200, y: 222, w: 360, h: 54, label: '2 · Expected Free Energy\nG(i,a) = epistemic + pragmatic', group: 'agent', emphasis: true },
    { id: 'sel', x: 200, y: 312, w: 360, h: 50, label: '3 · Action selection\n(i*, a*) = argmin G(i,a)', group: 'agent' },
    { id: 'exec', x: 120, y: 398, w: 230, h: 54, label: '4a · Execute\napply a* to feature i*', group: 'io' },
    { id: 'obs', x: 410, y: 398, w: 230, h: 54, label: '4b · Observe\nKL, activation, connectivity', group: 'io' },
    { id: 'learn', x: 200, y: 492, w: 360, h: 52, label: '5 · Online learning\nDirichlet update of A(o|s)', group: 'genmodel' },
  ],
  edges: [
    { from: 'A', to: 'infer' }, { from: 'B', to: 'infer' }, { from: 'C', to: 'infer' }, { from: 'D', to: 'infer' },
    { from: 'infer', to: 'efe' }, { from: 'efe', to: 'sel' },
    { from: 'sel', to: 'exec' }, { from: 'sel', to: 'obs' },
    { from: 'exec', to: 'learn' }, { from: 'obs', to: 'learn' },
    { from: 'learn', to: 'A', feedback: true, fromSide: 'left', toSide: 'left', curve: -150, label: 'update' },
  ],
};

/** ---- Fig 1 (paper): full ACD architecture ---- */
export const architecture: { nodes: FNode[]; edges: FEdge[]; groups: FGroup[]; viewW: number; viewH: number } = {
  viewW: 880,
  viewH: 560,
  groups: [
    { id: 'b', label: 'Attribution-graph backend', color: '#22d3ee', bbox: { x: 14, y: 30, w: 300, h: 510 } },
    { id: 'a', label: 'Active-Inference POMDP agent (pymdp)', color: '#a855f7', bbox: { x: 360, y: 30, w: 506, h: 510 } },
  ],
  nodes: [
    { id: 'p', x: 44, y: 62, w: 240, h: 46, label: 'Prompt  x', group: 'io', shape: 'round' },
    { id: 'llm', x: 44, y: 138, w: 240, h: 52, label: 'LLM with transcoders\n(Gemma-2-2B / Llama-3.2-1B)', group: 'backend' },
    { id: 'eap', x: 44, y: 222, w: 240, h: 50, label: 'circuit-tracer / EAP\n(edge attribution patching)', group: 'backend' },
    { id: 'graph', x: 44, y: 302, w: 240, h: 48, label: 'Attribution graph\n{(l,p,f)}, W, a', group: 'backend' },
    { id: 'prune', x: 44, y: 380, w: 240, h: 46, label: 'Pruning  (≥ 80% influence)', group: 'backend' },
    { id: 'cand', x: 44, y: 456, w: 240, h: 52, label: 'Candidate features\n+ importance imp(i)', group: 'backend' },

    { id: 'gA', x: 392, y: 70, w: 110, h: 44, label: 'A · P(o|s)', group: 'genmodel' },
    { id: 'gB', x: 512, y: 70, w: 110, h: 44, label: 'B · P(s′|s,a)', group: 'genmodel' },
    { id: 'gC', x: 392, y: 122, w: 110, h: 44, label: 'C · P(o)', group: 'genmodel' },
    { id: 'gD', x: 512, y: 122, w: 110, h: 44, label: 'D · P(s)', group: 'genmodel' },
    { id: 'infer', x: 470, y: 200, w: 320, h: 46, label: 'Variational inference  q(s|o)', group: 'agent' },
    { id: 'efe', x: 470, y: 270, w: 320, h: 46, label: 'EFE  G(i,a) = epistemic + pragmatic', group: 'agent', emphasis: true },
    { id: 'sel', x: 470, y: 340, w: 320, h: 46, label: 'Action selection  (i*, a*)', group: 'agent' },
    { id: 'interv', x: 470, y: 416, w: 320, h: 50, label: 'feature_intervention API\n(ablation / patching / steering)', group: 'io' },
    { id: 'kl', x: 470, y: 492, w: 320, h: 44, label: 'Intervened logits  +  KL divergence', group: 'io' },
  ],
  edges: [
    { from: 'p', to: 'llm' }, { from: 'llm', to: 'eap' }, { from: 'eap', to: 'graph' },
    { from: 'graph', to: 'prune' }, { from: 'prune', to: 'cand' },
    { from: 'cand', to: 'infer', fromSide: 'right', toSide: 'left', label: 'candidates' },
    { from: 'gA', to: 'infer' }, { from: 'gD', to: 'infer' },
    { from: 'infer', to: 'efe' }, { from: 'efe', to: 'sel' }, { from: 'sel', to: 'interv' },
    { from: 'interv', to: 'kl' },
    { from: 'kl', to: 'infer', feedback: true, fromSide: 'right', toSide: 'right', curve: 120, label: 'feedback' },
  ],
};

/** ---- Fig 3 (paper): the per-step decision loop ---- */
export const stepFlow: { nodes: FNode[]; edges: FEdge[]; groups: FGroup[]; viewW: number; viewH: number } = {
  viewW: 560,
  viewH: 880,
  groups: [],
  nodes: [
    { id: 'start', x: 190, y: 14, w: 180, h: 44, label: 'Start step t', group: 'io', shape: 'round' },
    { id: 'cand', x: 150, y: 86, w: 260, h: 54, label: 'EAP + prune graph →\ncandidate features {(l,p,f)} + imp(i)', group: 'backend' },
    { id: 'prior', x: 150, y: 168, w: 260, h: 48, label: 'Derive prior observation\nfrom imp(i)', group: 'agent' },
    { id: 'efe', x: 150, y: 244, w: 260, h: 48, label: 'Compute EFE G(i,a)\nfor all candidate–action pairs', group: 'agent', emphasis: true },
    { id: 'sel', x: 150, y: 320, w: 260, h: 44, label: 'Select (i*, a*) = argmin G(i,a)', group: 'agent' },
    { id: 'interv', x: 150, y: 392, w: 260, h: 50, label: 'Execute intervention\nfeature_intervention(l*,p*,f*)', group: 'io' },
    { id: 'measure', x: 150, y: 470, w: 260, h: 54, label: 'Measure KL divergence,\nactivation, connectivity', group: 'io' },
    { id: 'update', x: 150, y: 552, w: 260, h: 50, label: 'Update q(s), A(o|s),\nlayer priors', group: 'genmodel' },
    { id: 'check', x: 175, y: 628, w: 210, h: 84, label: 't < B and\nnot converged?', group: 'agent', shape: 'diamond' },
    { id: 'next', x: 360, y: 648, w: 150, h: 44, label: 'increment t', group: 'agent' },
    { id: 'stop', x: 175, y: 756, w: 210, h: 46, label: 'Stop — output circuit', group: 'io', shape: 'round', emphasis: true },
  ],
  edges: [
    { from: 'start', to: 'cand' }, { from: 'cand', to: 'prior' }, { from: 'prior', to: 'efe' },
    { from: 'efe', to: 'sel' }, { from: 'sel', to: 'interv' }, { from: 'interv', to: 'measure' },
    { from: 'measure', to: 'update' }, { from: 'update', to: 'check' },
    { from: 'check', to: 'next', fromSide: 'right', toSide: 'top', label: 'yes' },
    { from: 'next', to: 'start', feedback: true, fromSide: 'top', toSide: 'right', curve: 120 },
    { from: 'check', to: 'stop', label: 'no' },
  ],
};
