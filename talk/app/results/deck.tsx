'use client';

import Link from 'next/link';
import Deck, { Slide, Reveal, NextLead } from '../../components/Deck';
import { Eq, Callout } from '../../components/Prose';
import {
  EffBars, CumKLChart, RCKChart, SteeringChart, ActionStrip, AConvChart, DomainTable, Legend,
} from '../../components/Charts';
import { OracleRaceSVG, LayerBinsSVG, ResultsMapSVG } from '../../components/TeachResults';
import { ExploreExploitSVG } from '../../components/TeachBridge';

function Num({ children }: { children: React.ReactNode }) {
  return <td className="num">{children}</td>;
}

export default function ResultsDeck() {
  return (
    <Deck part="Chapter III · The evidence" title="Results, in full">
      {/* 1 — chapter map */}
      <Slide
        title="How to read this chapter"
        brief="Roadmap: protocol → headline → the control → steering → the failure → verdict"
        steps={1}
        notes={
          <>
            <p>
              This chapter reads in a specific order, and the order is protective. The scoring
              protocol comes first, because without it every later number can be misread. Then the
              headline result and its per-step dynamics; then the control that keeps the most
              spectacular number from being oversold; then a steering experiment with a null the
              paper chose to keep; then the worst result in the paper, diagnosed in model terms;
              and finally the two-column verdict of what survives. Every chart is rendered from
              the repository&rsquo;s raw JSON at build time — nothing redrawn by hand.
            </p>
          </>
        }
      >
        <p className="kicker">Chapter III · the evidence</p>
        <h1>
          Results, <span className="accent">in a protective order</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <ResultsMapSVG stage={6} />
        </div>
        <Reveal at={1}>
          <div className="take teal">
            Every number on the next slides traces to a JSON file in the public repository — and
            every chart is <strong>rendered from those files at build time</strong>.
          </div>
        </Reveal>
        <NextLead>First, the scoring rule — it carries the weight.</NextLead>
      </Slide>

      {/* 2 — protocol */}
      <Slide
        title="The protocol: race an oracle"
        brief="Ablation-only, scored as % of an oracle that knows every KL — capped at 100 by construction"
        steps={2}
        notes={
          <>
            <p>
              Every headline metric is <strong>bounded oracle efficiency</strong>: all selectors
              restricted to the same action (ablation), scored against an ablation oracle that
              knows every true KL in advance. The ratio cannot exceed 100% — the bound is
              structural. Comparisons where the agent may choose intervention types live on a
              separate, explicitly-labelled scale (RCK, later in this chapter). Budget B = 20
              probes per prompt; bootstrap 95% CIs over prompts; significance by exact paired
              permutation test.
            </p>
            <Eq tex={String.raw`\eta \;=\; \frac{\mathrm{CumKL}^{\text{abl}}_{\text{method}}}{\mathrm{CumKL}^{\text{abl}}_{\text{oracle}}}\times 100\% \;\in\; [0,100]\%`} />
          </>
        }
      >
        <p className="kicker">III.1 · The scoring rule</p>
        <h1>
          Everyone races <span className="accent">an oracle</span>
        </h1>
        <div className="fig-panel" style={{ maxWidth: 1000 }}>
          <OracleRaceSVG />
        </div>
        <ul className="pts compact" style={{ marginTop: '0.7rem' }}>
          <Reveal at={1}>
            <li>
              Same 20 ablations for everyone, scored as <strong>% of an oracle that already knows
              every answer</strong> — capped at 100 by construction
            </li>
          </Reveal>
          <Reveal at={2}>
            <li>
              Bootstrap 95% CIs · exact paired permutation tests · steering comparisons kept on a{' '}
              <em>separate, labelled scale</em>
            </li>
          </Reveal>
        </ul>
        <NextLead>With the rule fixed, the headline can be read safely.</NextLead>
      </Slide>

      {/* 3 — headline */}
      <Slide
        title="Headline: beats random, loses to EAP"
        brief="Gemma IOI: 82% of oracle, +43.5% vs random (p = .031) — EAP's static ranking stays ahead"
        steps={3}
        notes={
          <>
            <div className="tbl-wrap">
              <table className="tbl">
                <caption>IOI, fair ablation-only protocol, 5 prompts, B = 20. Bounded oracle efficiency with bootstrap 95% CIs (paper Table 3).</caption>
                <thead>
                  <tr><th>Model</th><th>Method</th><th className="num">Mean KL</th><th className="num">Oracle eff. [95% CI]</th></tr>
                </thead>
                <tbody>
                  <tr><td rowSpan={6}>Gemma-2-2B</td><td>EAP</td><Num>0.000793</Num><Num>91.9 [82.1, 96.5]</Num></tr>
                  <tr className="hl"><td>POMDP-abl</td><Num>0.000707</Num><Num>82.0 [73.8, 87.8]</Num></tr>
                  <tr><td>Bandit</td><Num>0.000642</Num><Num>74.4 [67.9, 79.0]</Num></tr>
                  <tr><td>UCB</td><Num>0.000635</Num><Num>73.7 [59.0, 82.2]</Num></tr>
                  <tr><td>Greedy</td><Num>0.000577</Num><Num>66.9 [47.5, 83.0]</Num></tr>
                  <tr><td>Random</td><Num>0.000493</Num><Num>57.1 [49.0, 60.2]</Num></tr>
                  <tr><td rowSpan={6}>Llama-3.2-1B</td><td>EAP</td><Num>0.009291</Num><Num>95.2 [87.0, 97.7]</Num></tr>
                  <tr><td>Bandit</td><Num>0.008606</Num><Num>88.2 [79.0, 96.7]</Num></tr>
                  <tr><td>UCB</td><Num>0.005389</Num><Num>55.2 [10.5, 92.7]</Num></tr>
                  <tr className="hl"><td>POMDP-abl</td><Num>0.005086</Num><Num>52.1 [17.4, 90.2]</Num></tr>
                  <tr><td>Random</td><Num>0.004720</Num><Num>48.4 [31.6, 64.7]</Num></tr>
                  <tr><td>Greedy</td><Num>0.003982</Num><Num>40.8 [11.7, 74.1]</Num></tr>
                </tbody>
              </table>
            </div>
            <p>
              On Gemma the agent reaches 82.0% of an oracle that already knows every answer —
              +43.5% relative over random. The exact paired permutation test gives p = 0.031, the
              minimum attainable two-condition p at n = 5: the agent beat random on{' '}
              <em>every single prompt</em>. It also edges the engineered bandit (74.4%) and UCB
              (73.7%). On Llama it does not: 52.1% with a CI touching random. On both models,
              EAP&rsquo;s static ranking (91.9 / 95.2%) is untouched — a fact the paper states in
              its abstract, and which the end of this chapter explains structurally.
            </p>
          </>
        }
      >
        <p className="kicker">III.2 · Headline — IOI, Gemma-2-2B</p>
        <h1>
          Beats random where it matters.
          <br />
          <span className="accent">Loses to EAP everywhere.</span>
        </h1>
        <div className="stats">
          <Reveal at={1}>
            <div className="stat">
              <div className="v">82.0%</div>
              <div className="k">of an oracle that knows every answer</div>
            </div>
          </Reveal>
          <Reveal at={2}>
            <div className="stat">
              <div className="v">+43.5%</div>
              <div className="k">vs random · p = 0.031 — won on every prompt</div>
            </div>
          </Reveal>
          <Reveal at={3}>
            <div className="stat">
              <div className="v amber">91.9%</div>
              <div className="k">EAP&rsquo;s static ranking — still champion</div>
            </div>
          </Reveal>
        </div>
        <Reveal at={3}>
          <div className="take">
            On Llama: 52.1%, CI touching random. Both facts sit in the paper&rsquo;s abstract —
            full table in the notes.
          </div>
        </Reveal>
        <NextLead>The same story, seen across all four task × model cells.</NextLead>
      </Slide>

      {/* 4 — all cells */}
      <Slide
        title="All four cells at once"
        brief="EAP always first; the agent's rank degrades from Gemma to Llama"
        steps={1}
        notes={
          <>
            <p>
              Bounded oracle efficiency across all four task×model cells, with bootstrap 95% CIs —
              rendered from <code>stats.json</code>. Two patterns to narrate: EAP is always first,
              and the agent&rsquo;s rank degrades from Gemma to Llama — catastrophically so on
              multi-step, which becomes the diagnosed failure later in this chapter.
            </p>
          </>
        }
      >
        <p className="kicker">III.2 · The wide shot</p>
        <h2>
          Two patterns: <span className="dim">EAP always first · the teal bar slides toward Llama</span>
        </h2>
        <div className="fig-panel">
          <EffBars />
          <Legend keys={['eap', 'ai_abl', 'bandit', 'ucb', 'greedy', 'random']} />
        </div>
        <Reveal at={1}>
          <div className="take teal">
            The teal bar&rsquo;s slide is the arc of this chapter — it ends at 9.3%, and the
            diagnosis is the most instructive result in the paper.
          </div>
        </Reveal>
        <NextLead>Zoom into the race, step by step.</NextLead>
      </Slide>

      {/* 5 — per-step dynamics */}
      <Slide
        title="Per-step dynamics"
        brief="Nobody crosses the oracle envelope — the bound is structural, not a failure"
        notes={
          <>
            <p>
              Mean cumulative ablation KL per step (IOI). No method crosses the oracle envelope
              (dashed) — the bound is structural, since all methods share the same action. Gemma:
              the agent tracks EAP closely from around step 6. Llama: the agent sits with
              greedy/random from early on. Reading tip for a live audience: the vertical gap
              between a curve and the dashed envelope is the price of not knowing the answers in
              advance.
            </p>
          </>
        }
      >
        <p className="kicker">III.2 · Step by step</p>
        <h2>
          Nobody crosses the envelope — <span className="dim">the gap is the price of not knowing</span>
        </h2>
        <div className="fig-panel">
          <CumKLChart task="ioi" />
          <Legend keys={['oracle', 'eap', 'ai_abl', 'bandit', 'greedy']} />
        </div>
        <NextLead>Now free the agent&rsquo;s choice of lever — and watch a number explode.</NextLead>
      </Slide>

      {/* 6 — RCK */}
      <Slide
        title="1255% — and the control that tames it"
        brief="Action-matched baselines show the steering lever, not intelligence, does most of the amplifying"
        steps={3}
        notes={
          <>
            <p>
              Free the agent to choose intervention <em>types</em> and its cumulative KL explodes
              past the ablation oracle — 1255% on Gemma IOI. It would have been easy to print that
              as the headline. The paper instead builds the control that dismantles it:{' '}
              <strong>action-matched baselines</strong>. Give the same steering lever to random
              selection: 888%. To the bandit: 1185%. Most of the amplification is the lever
              itself. The full agent does top the Gemma ordering — EFE selection adds a real
              margin — but the paper&rsquo;s wording is exact: RCK “must not be read as a
              super-oracle discovery rate.”
            </p>
            <div className="tbl-wrap">
              <table className="tbl">
                <caption>Relative Cumulative KL vs the ablation oracle (=100%), bootstrap 95% CIs (paper Table 4).</caption>
                <thead>
                  <tr><th>Model</th><th>Method</th><th className="num">IOI RCK</th><th className="num">Multi-step RCK</th></tr>
                </thead>
                <tbody>
                  <tr className="hl"><td rowSpan={5}>Gemma-2-2B</td><td>POMDP (multi)</td><Num>1255 [974, 1671]</Num><Num>979 [792, 1374]</Num></tr>
                  <tr><td>Bandit+steer</td><Num>1185 [780, 1797]</Num><Num>1209 [798, 1935]</Num></tr>
                  <tr><td>Random+steer</td><Num>888 [672, 1182]</Num><Num>737 [496, 1073]</Num></tr>
                  <tr><td>Greedy+steer</td><Num>881 [605, 1675]</Num><Num>1223 [707, 1907]</Num></tr>
                  <tr><td>Random-action</td><Num>371 [264, 528]</Num><Num>249 [213, 296]</Num></tr>
                  <tr><td rowSpan={5}>Llama-3.2-1B</td><td>Bandit+steer</td><Num>1467 [329, 2848]</Num><Num>279 [203, 412]</Num></tr>
                  <tr><td>Random+steer</td><Num>961 [181, 1907]</Num><Num>381 [115, 868]</Num></tr>
                  <tr><td>Random-action</td><Num>429 [86, 811]</Num><Num>164 [68, 340]</Num></tr>
                  <tr><td>Greedy+steer</td><Num>353 [244, 482]</Num><Num>140 [55, 293]</Num></tr>
                  <tr className="hl"><td>POMDP (multi)</td><Num>91 [27, 320]</Num><Num>34 [19, 56]</Num></tr>
                </tbody>
              </table>
            </div>
          </>
        }
      >
        <p className="kicker">III.3 · The control</p>
        <h1>
          Amplification is the <span className="accent">lever</span>,
          <br />
          not the intelligence
        </h1>
        <div className="stats">
          <Reveal at={1}>
            <div className="stat">
              <div className="v">1255%</div>
              <div className="k">agent free to steer, vs the ablation oracle — a tempting headline</div>
            </div>
          </Reveal>
          <Reveal at={2}>
            <div className="stat">
              <div className="v amber">888%</div>
              <div className="k">random selection, same steering lever</div>
            </div>
          </Reveal>
          <Reveal at={2}>
            <div className="stat">
              <div className="v amber">1185%</div>
              <div className="k">bandit, same lever</div>
            </div>
          </Reveal>
        </div>
        <Reveal at={3}>
          <div className="take">
            The lever does most of the amplifying. The agent still tops the ordering — but the
            paper builds its own deflating control and prints the exact caveat.
          </div>
        </Reveal>
        <NextLead>The same table hides the single best exhibit in the paper — on Llama.</NextLead>
      </Slide>

      {/* 7 — the Llama reversal */}
      <Slide
        title="The Llama reversal"
        brief="Same objective, same priors — the agent declines to steer: RCK 91%, 70/100 ablations"
        steps={2}
        notes={
          <>
            <p>
              RCK on a log scale, agent vs action-matched baselines, IOI. Amber dashes mark the
              ablation oracle at 100%. On Gemma, everything with steering blows past 100 — the
              lever dominates and the agent leads. On Llama, the agent lands <em>below</em> 100 —
              at 91% — because it mostly declined to steer: 70 of its 100 actions were ablations.
            </p>
            <p>
              Same objective, same priors, different observation statistics → different emergent
              policy. A hard-coded “steer for the KL” heuristic cannot produce that reversal; a
              belief-driven policy can. This is the single best exhibit that the action selection
              is inference, not schedule — exactly the model-sensitivity Chapter II predicted.
            </p>
          </>
        }
      >
        <p className="kicker">III.3 · Same objective, two policies</p>
        <div className="cols cols-60">
          <div className="fig-panel">
            <RCKChart />
          </div>
          <div>
            <h2>The reversal</h2>
            <ul className="pts compact">
              <Reveal at={1}>
                <li>
                  On Llama the agent&rsquo;s RCK is <strong>91%</strong> — it{' '}
                  <em>declined to steer</em> (70/100 ablations)
                </li>
              </Reveal>
              <Reveal at={2}>
                <li>
                  Same objective, same priors — <strong>different observations, different
                  emergent policy</strong>
                </li>
              </Reveal>
            </ul>
            <Reveal at={2}>
              <div className="take teal">
                A hard-coded “steer for KL” rule cannot do this. <strong>Inference can.</strong>
              </div>
            </Reveal>
          </div>
        </div>
        <NextLead>Chapter II drew a prediction as two crossing curves. Time to check it.</NextLead>
      </Slide>

      {/* 8 — prediction vs observation */}
      <Slide
        title="Prediction, meet data"
        brief="The predicted ablate→steer handoff appears on Gemma — and rationally fails to on Llama"
        steps={2}
        notes={
          <>
            <p>
              Top: the prediction from Chapter II — epistemic value decays as beliefs sharpen,
              pragmatic value takes over, so early ablations should hand off to steering, with no
              schedule coded anywhere. Bottom: every action the agent actually took — 5 prompts ×
              20 steps per model, from the raw run logs. Gemma: one exploratory ablation opens
              each prompt, then steering dominates (94/100 steps; 1 patch). Llama: ablation-heavy
              throughout (70/100), because noisier observation statistics keep beliefs — and hence
              epistemic value — from collapsing.
            </p>
            <p>
              Both patterns are the same objective responding to different environments. That is
              the difference between a schedule and an inference, made visible in a single figure
              pair. One honest reading note: the prediction is of the <em>order</em> (ablate era
              → steer era), not the timing — and on Gemma the handoff is abrupt (one opening
              ablation), consistent with the κ-seeded prior collapsing beliefs quickly. The
              discriminating evidence is the pair: handoff on Gemma, rational refusal to hand off
              on Llama.
            </p>
          </>
        }
      >
        <p className="kicker">III.4 · The falsifiable bit</p>
        <h1>
          The handoff, <span className="accent">observed</span>
        </h1>
        <Reveal at={1}>
          <div className="fig-panel" style={{ maxWidth: 1000 }}>
            <ExploreExploitSVG />
          </div>
        </Reveal>
        <Reveal at={2}>
          <div className="fig-panel" style={{ maxWidth: 1000, marginTop: '0.7rem' }}>
            <ActionStrip />
          </div>
        </Reveal>
        <NextLead>Next: the Golden Gate experiment from Chapter I — quantified, with a control.</NextLead>
      </Slide>

      {/* 9 — steering */}
      <Slide
        title="Steering: strong dose-response, a kept null"
        brief="Manipulability confirmed (p < 10⁻⁸); selectivity vs random features — not significant"
        steps={2}
        notes={
          <>
            <p>
              The Golden-Gate-style test: 5 concept prompts, top-10 circuit-selected features
              each, multipliers m ∈ {'{'}0, 1.5, 2, 3, 5, 10{'}'} — against a matched
              random-feature control. Manipulability (H2a) is unambiguous: prediction changes rise
              monotonically with dose, 11/50 (Gemma) and 9/50 (Llama) at m = 10, binomial
              p &lt; 10⁻⁸ against chance. Mean KL of selected features at m = 10 is 4× the
              control&rsquo;s (0.367 vs 0.086 on Gemma), and steered outputs amplify coherent
              concepts rather than collapsing to function words.
            </p>
            <p>
              But selectivity (H2b) — are <em>circuit-selected</em> features more steerable than
              random active ones? — fails: 11 vs 8 and 9 vs 6 changes, Fisher exact n.s., at
              multipliers that are frankly off-distribution. The paper prints the null with the
              OOD caveat. For an expert audience the null is informative in itself: it
              independently reproduces the steering-vector reliability critique from the
              interpretability literature (large multipliers ≈ generic activation scaling) — with
              the control the original Golden Gate demo never had.
            </p>
          </>
        }
      >
        <p className="kicker">III.5 · The Golden Gate test, quantified</p>
        <div className="cols cols-60">
          <div className="fig-panel">
            <SteeringChart />
          </div>
          <div>
            <h2>Two verdicts</h2>
            <ul className="pts compact">
              <Reveal at={1}>
                <li>
                  <strong>Dose-response: yes</strong> — monotone in multiplier, p &lt; 10⁻⁸;
                  selected-feature KL 4× the control&rsquo;s
                </li>
              </Reveal>
              <Reveal at={2}>
                <li>
                  <strong>Selectivity: no</strong> — circuit-selected vs random-active, Fisher
                  exact n.s.
                </li>
              </Reveal>
            </ul>
            <Reveal at={2}>
              <div className="take">
                The null stays on the slide: it reproduces the steering-reliability critique —
                with the control the original demo never had.
              </div>
            </Reveal>
          </div>
        </div>
        <NextLead>Now the worst number in the paper — and why it is the most instructive.</NextLead>
      </Slide>

      {/* 10 — the failure */}
      <Slide
        title="The failure: 9.3%, diagnosed"
        brief="Llama multi-step: a 3-bin state space cannot see early-layer mass; 6 bins → 37.8%"
        steps={3}
        notes={
          <>
            <div className="tbl-wrap">
              <table className="tbl">
                <caption>Multi-step reasoning, fair protocol, 3 prompts (paper Table 6, abridged).</caption>
                <thead>
                  <tr><th>Model</th><th>Method</th><th className="num">Oracle eff. [95% CI]</th></tr>
                </thead>
                <tbody>
                  <tr><td rowSpan={3}>Gemma-2-2B</td><td>EAP</td><Num>98.1 [97.2, 99.6]</Num></tr>
                  <tr className="hl"><td>POMDP-abl</td><Num>73.0 [63.6, 84.2]</Num></tr>
                  <tr><td>Random</td><Num>56.2 [52.4, 57.4]</Num></tr>
                  <tr><td rowSpan={4}>Llama-3.2-1B</td><td>EAP</td><Num>97.6 [62.8, 99.4]</Num></tr>
                  <tr><td>Random</td><Num>44.8 [39.1, 54.5]</Num></tr>
                  <tr className="hl"><td>POMDP-abl</td><Num>9.3 [4.4, 55.9]</Num></tr>
                  <tr className="hl"><td>&nbsp;&nbsp;+ six layer-role bins</td><Num>37.8</Num></tr>
                </tbody>
              </table>
            </div>
            <p>
              Below random. The per-prompt diagnosis: 4.4% and 15.2% on the two prompts whose
              oracle KL mass sits in <em>early layers</em> (oracle CumKL 0.461, 0.257); recovery
              to 55.9% on the one late-layer prompt (0.016). The mechanism: with 16 layers cut
              into three role bins, functionally distinct early-middle layers collapse into one
              hidden state — the belief state <em>cannot represent</em> where the causal mass is.
              The targeted fix — six bins, one flag — quadruples efficiency to 37.8% without
              touching the EFE objective. Still behind the bandit; reported anyway. Caveat kept
              with it: the fix was chosen by a human post-hoc, not discovered by the agent —
              structure learning remains future work.
            </p>
          </>
        }
      >
        <p className="kicker">III.6 · The worst number, kept</p>
        <h1>
          <span className="accent">9.3%.</span> Below random.
        </h1>
        <Reveal at={1}>
          <div className="fig-panel" style={{ maxWidth: 1020 }}>
            <LayerBinsSVG />
          </div>
        </Reveal>
        <Reveal at={2}>
          <div className="take violet">
            The structure-learning moral this audience would predict: when the generative model is
            too coarse, refine the <strong>model</strong>, not the objective. (Caveat kept: the
            fix was human, post-hoc — the agent did not find it.)
          </div>
        </Reveal>
        <Reveal at={3}>
          <NextLead>What a state-space failure looks like as a curve.</NextLead>
        </Reveal>
      </Slide>

      {/* 11 — multistep chart */}
      <Slide
        title="A state-space failure, as a curve"
        brief="The teal curve flatlines against an oracle whose mass it cannot represent"
        notes={
          <>
            <p>
              Multi-step cumulative KL. Gemma: the agent is respectable at 73%. Llama: the teal
              curve flatlines against an oracle whose mass it cannot see — the visual signature of
              a state-space failure, as opposed to an objective failure: the agent is optimising
              correctly over a representation that cannot express the answer.
            </p>
          </>
        }
      >
        <p className="kicker">III.6 · The signature</p>
        <h2>
          The teal curve flatlines — <span className="dim">it cannot represent where the mass is</span>
        </h2>
        <div className="fig-panel">
          <CumKLChart task="multistep" />
          <Legend keys={['oracle', 'eap', 'ai_abl', 'bandit', 'greedy']} />
        </div>
        <NextLead>Two quieter results with scientific content of their own.</NextLead>
      </Slide>

      {/* 12 — geography + learning */}
      <Slide
        title="Geography, and a likelihood that learns"
        brief="Facts live late, reasoning early — on both architectures; the learned A converges"
        steps={2}
        notes={
          <>
            <p>
              Across five cognitive domains × two models the agent is a reliable selector (77–94%
              on Gemma, 57–98% on Llama — above random everywhere, trading wins with
              bandit/greedy, below EAP). The finding with scientific content is the{' '}
              <strong>layer geography</strong>: factual domains (geography, history) put their
              top-10 causal mass in late layers; reasoning domains (logic, math) recruit early
              layers; science sits mixed — replicated across both architectures, consistent with
              the ROME/factual-recall picture. IOI&rsquo;s single highest-KL features sit at
              layers 24–25 of Gemma&rsquo;s 26 (e.g. <code>L25_P14_F4717</code>) — the late-layer
              name-mover signature from Chapter I, recovered at transcoder resolution.
            </p>
            <p>
              The Dirichlet-learned likelihood is not decoration: mean per-step L1 drift of the
              KL-likelihood matrix halves over the budget (1.03×10⁻³ → 5.0×10⁻⁴ on Gemma IOI) and
              belief entropy falls monotonically.
            </p>
            <h3>Sensitivity — one knob matters, and it is not an agent knob</h3>
            <p>
              Threshold sweep (Gemma IOI, baseline 82.0%): activation thresholds ±20% → 86.2 /
              80.3%; KL thresholds +20% → 82.0%; KL thresholds <strong>−20% → 60.2%</strong>.
              Tightening the KL bins injects observation noise into every belief update — the
              discretisation is the sensitive interface between continuous environment and
              discrete agent. The agent&rsquo;s own hyperparameters are flat: γ ∈ {'{'}4, 16,
              64{'}'}, η ∈ {'{'}0.5, 1, 2{'}'}, pragmatic weight ∈ {'{'}0.5, 1, 2{'}'} all leave
              Gemma IOI at 82.0%. The model&rsquo;s real prior lives where Chapter II said it was:
              in the discretisation, not in γ.
            </p>
            <DomainTable />
          </>
        }
      >
        <p className="kicker">III.7 · Two quiet results</p>
        <div className="cols cols-60">
          <div>
            <h1>
              Circuits have
              <br />
              <span className="accent">geography</span>
            </h1>
            <ul className="pts compact">
              <Reveal at={1}>
                <li>
                  <strong>Facts live late</strong> (geography, history) · <strong>reasoning
                  recruits early</strong> (logic, math) — replicated on both architectures
                </li>
              </Reveal>
              <Reveal at={2}>
                <li>
                  And the likelihood <strong>actually learns</strong>: per-step drift of A halves
                  over the 20-step budget
                </li>
              </Reveal>
            </ul>
          </div>
          <div className="fig-panel">
            <AConvChart />
          </div>
        </div>
        <NextLead>Every hypothesis was registered before the runs. Time to grade them.</NextLead>
      </Slide>

      {/* 13 — scorecard */}
      <Slide
        title="Scorecard: graded in public"
        brief="Four pre-registered hypotheses — two rejections, two partials, printed in the paper"
        steps={1}
        notes={
          <>
            <p>
              Two rejections and two partials, printed in the paper&rsquo;s own hypothesis table
              (Table 10). Presentational value for any speaker using this material: when an expert
              audience probes the weak spots, the answer is already on the slide — which converts
              a gotcha into a discussion.
            </p>
          </>
        }
      >
        <p className="kicker">III.8 · The grade sheet</p>
        <h2>Four hypotheses, pre-registered</h2>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead>
              <tr><th>Hypothesis</th><th>Result</th><th className="num">p</th><th>Verdict</th></tr>
            </thead>
            <tbody>
              <tr><td>H1 · Efficiency vs random</td><td>+43.5% / +7.8%</td><Num>0.031 / 0.41</Num><td>Gemma ✓ · <strong>Llama ✗</strong></td></tr>
              <tr><td>H2a · Manipulability</td><td>11/50 &amp; 9/50 at m=10</td><Num>&lt;10⁻⁸</Num><td>✓</td></tr>
              <tr><td>H2b · Selectivity</td><td>11 vs 8; 9 vs 6</td><Num>n.s.</Num><td><strong>✗</strong></td></tr>
              <tr><td>H3 · Discovery ≥ 50%</td><td>82.0 / 52.1 · 73.0 / 9.3</td><Num>—</Num><td>✓ except <strong>Llama multi-step</strong></td></tr>
              <tr><td>H4 · Transfer</td><td>Gemma full, Llama partial</td><Num>—</Num><td><strong>Partial</strong></td></tr>
            </tbody>
          </table>
        </div>
        <Reveal at={1}>
          <div className="take">
            Two rejections, two partials — in the paper&rsquo;s own table. Weak spots on the
            slide are weak spots the audience cannot spring.
          </div>
        </Reveal>
        <NextLead>So — after all of it — what survives?</NextLead>
      </Slide>

      {/* 14 — what survives */}
      <Slide
        title="Verdict: what survives"
        brief="Established vs not — and why EAP winning was structurally likely"
        steps={3}
        notes={
          <>
            <p>
              <strong>Established:</strong> EFE-guided selection significantly beats random on
              Gemma IOI with the strongest available test at n = 5; the agent is competitive with
              engineered heuristics on Gemma across IOI and all five domains; the
              exploration→exploitation and architecture-dependent action policies emerge from the
              generative model; the learned likelihood converges; transcoder features are
              dose-dependently causal; task-dependent layer geography replicates across two
              architectures. <strong>Not established:</strong> superiority over EAP anywhere;
              steering selectivity of circuit-selected features; transfer of the selection
              advantage to the shallow model; multi-step competence under the 3-bin state space.
            </p>
            <Callout title="Why EAP winning was structurally likely" tone="amber">
              The candidate set was drawn from the top of the attribution graph, and EAP ranks by
              direct-to-logit attribution <em>on that same graph</em> — the evaluation is
              selection efficiency within EAP&rsquo;s own shortlist, its home turf. The agent
              starts from a κ-scaled version of those importances as a mere prior and must
              rediscover the ordering through 20 noisy, discretised observations. That EAP is hard
              to out-rank in-distribution is the expected result; the agent&rsquo;s additions —
              uncertainty, multi-action choice, online learning — pay off precisely where a
              static list cannot adapt: auditing under distribution shift, multi-prompt budgets,
              intervention types with different costs. That frontier is future work, and a fair
              question for any audience.
            </Callout>
          </>
        }
      >
        <p className="kicker">III.9 · The verdict</p>
        <h1>What survives</h1>
        <div className="cols">
          <div>
            <Reveal at={1}>
              <h2 style={{ color: 'var(--teal-bright)' }}>Established</h2>
              <ul className="pts compact">
                <li>Beats random on Gemma IOI — strongest test available at n = 5</li>
                <li>Explore→exploit and per-architecture policies <em>emerge</em> — no schedule</li>
                <li>The likelihood converges; features are dose-dependently causal</li>
              </ul>
            </Reveal>
          </div>
          <div>
            <Reveal at={2}>
              <h2 style={{ color: 'var(--amber)' }}>Not established</h2>
              <ul className="pts compact">
                <li>Superiority over EAP — anywhere</li>
                <li>Steering selectivity of circuit-selected features</li>
                <li>Transfer of the advantage to the shallow model</li>
              </ul>
            </Reveal>
          </div>
        </div>
        <Reveal at={3}>
          <div className="take" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span>
              The value is not raw efficiency — a static ranking wins that. It is a principled,
              adaptive, <strong>self-diagnosing policy</strong> for spending causal budget.
            </span>
            <Link
              href="/demo"
              style={{
                fontFamily: 'var(--grotesk)', fontWeight: 600, fontSize: '.9rem', textDecoration: 'none',
                background: 'var(--teal)', color: '#fff', padding: '10px 20px', borderRadius: 999, whiteSpace: 'nowrap',
              }}
            >
              Chapter IV: watch it run →
            </Link>
          </div>
        </Reveal>
      </Slide>
    </Deck>
  );
}
