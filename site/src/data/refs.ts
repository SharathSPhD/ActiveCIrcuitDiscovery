import type { Reference } from '../components/ui/CitationList.astro';

// Citations verified against the paper's own bibliography (paper/references.bib).
export const REFS: Record<string, Reference> = {
  Friston2010: { authors: 'Friston, K. J.', year: '2010', title: 'The free-energy principle: a unified brain theory?', venue: 'Nature Reviews Neuroscience', url: 'https://doi.org/10.1038/nrn2787' },
  Friston2017: { authors: 'Friston, K. J. et al.', year: '2017', title: 'Active inference: a process theory', venue: 'Neural Computation', url: 'https://doi.org/10.1162/NECO_a_00912' },
  DaCosta2020: { authors: 'Da Costa, L. et al.', year: '2020', title: 'Active inference on discrete state-spaces: a synthesis', venue: 'Journal of Mathematical Psychology', url: 'https://doi.org/10.1016/j.jmp.2020.102447' },
  Parr2022: { authors: 'Parr, T., Pezzulo, G. & Friston, K. J.', year: '2022', title: 'Active Inference: The Free Energy Principle in Mind, Brain, and Behavior', venue: 'MIT Press' },
  Pymdp2022: { authors: 'Heins, C. et al.', year: '2022', title: 'pymdp: A Python library for active inference in discrete state spaces', venue: 'Journal of Open Source Software', url: 'https://doi.org/10.21105/joss.04098' },
  MacKay2003: { authors: 'MacKay, D. J. C.', year: '2003', title: 'Information Theory, Inference, and Learning Algorithms', venue: 'Cambridge University Press', url: 'https://www.inference.org.uk/itila/' },
  Olah2020: { authors: 'Olah, C. et al.', year: '2020', title: 'Zoom In: An Introduction to Circuits', venue: 'Distill', url: 'https://doi.org/10.23915/distill.00024.001' },
  Elhage2021: { authors: 'Elhage, N. et al.', year: '2021', title: 'A Mathematical Framework for Transformer Circuits', venue: 'Transformer Circuits Thread', url: 'https://transformer-circuits.pub/2021/framework/index.html' },
  Wang2022: { authors: 'Wang, K. et al.', year: '2023', title: 'Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small', venue: 'ICLR', url: 'https://openreview.net/forum?id=NpsVSN6o4ul' },
  Conmy2023: { authors: 'Conmy, A. et al.', year: '2023', title: 'Towards Automated Circuit Discovery for Mechanistic Interpretability', venue: 'NeurIPS', url: 'https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html' },
  Syed2023: { authors: 'Syed, A., Rager, C. & Conmy, A.', year: '2023', title: 'Attribution Patching Outperforms Automated Circuit Discovery', venue: 'arXiv:2310.10348', url: 'https://arxiv.org/abs/2310.10348' },
  Cunningham2024: { authors: 'Cunningham, H. et al.', year: '2024', title: 'Sparse Autoencoders Find Highly Interpretable Features in Language Models', venue: 'ICLR', url: 'https://openreview.net/forum?id=F76bwRSLeK' },
  Bricken2023: { authors: 'Bricken, T. et al.', year: '2023', title: 'Towards Monosemanticity: Decomposing Language Models With Dictionary Learning', venue: 'Transformer Circuits Thread', url: 'https://transformer-circuits.pub/2023/monosemanticity/index.html' },
  GemmaScope2024: { authors: 'Lieberum, T. et al.', year: '2024', title: 'Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2', venue: 'arXiv:2408.05147', url: 'https://arxiv.org/abs/2408.05147' },
  Ameisen2025: { authors: 'Ameisen, E. et al.', year: '2025', title: 'Circuit Tracing: Revealing Computational Graphs in Language Models', venue: 'Transformer Circuits Thread', url: 'https://transformer-circuits.pub/2025/attribution-graphs/methods.html' },
  Lindsey2025: { authors: 'Lindsey, J. et al.', year: '2025', title: 'On the Biology of a Large Language Model', venue: 'Transformer Circuits Thread', url: 'https://transformer-circuits.pub/2025/attribution-graphs/biology.html' },
  team2024gemma: { authors: 'Gemma Team', year: '2024', title: 'Gemma: Open Models Based on Gemini Research and Technology', venue: 'arXiv:2403.08295', url: 'https://arxiv.org/abs/2403.08295' },
  Dubey2024llama: { authors: 'Dubey, A. et al.', year: '2024', title: 'The Llama 3 Herd of Models', venue: 'arXiv:2407.21783', url: 'https://arxiv.org/abs/2407.21783' },
};

// Ordered list for the "Further Reading" section.
export const READING_ORDER: (keyof typeof REFS)[] = [
  'Friston2010', 'Friston2017', 'DaCosta2020', 'Parr2022', 'Pymdp2022', 'MacKay2003',
  'Olah2020', 'Elhage2021', 'Wang2022', 'Conmy2023', 'Syed2023',
  'Cunningham2024', 'Bricken2023', 'GemmaScope2024', 'Ameisen2025', 'Lindsey2025',
  'team2024gemma', 'Dubey2024llama',
];
