import type { Metadata } from 'next';
import Link from 'next/link';
import { Kicker, Callout, Deep, Fig, Eq } from '../../components/Prose';
import { EffBars, CumKLChart, RCKChart, SteeringChart, ActionStrip, AConvChart, DomainTable, Legend } from '../../components/Charts';

export const metadata: Metadata = {
  title: 'Part III — Results, in Full and Without Varnish',
  description:
    'Bounded oracle efficiency, the RCK decomposition, the steering selectivity null, the Llama multi-step failure, sensitivity sweeps, and the hypothesis scorecard.',
};

function Num({ children }: { children: React.ReactNode }) {
  return <td className="num">{children}</td>;
}

export default function Page() {
  return (
    <>
      <section className="band dark" style={{ paddingTop: '4.5rem' }}>
        <div className="reading">
          <Kicker>Part III · ~15 minutes</Kicker>
          <h2 className="sec">Results, in full and without varnish</h2>
          <p className="lede">
            The protocol before the numbers, because the protocol carries the honesty: every
            headline metric is <strong>bounded oracle efficiency</strong> — all selectors restricted
            to ablation, scored against an ablation oracle that knows every true KL in advance, so
            the ratio cannot exceed 100%. Steering-enabled comparisons live on a separate,
            explicitly-labelled scale (RCK). Budget B = 20 per prompt; bootstrap 95% CIs over
            prompts; significance by exact paired permutation test. Every number below is traceable
            to a JSON file in the public repo — and every chart on this page is rendered from those
            JSONs at build time, not redrawn by hand.
          </p>
          <Eq tex={String.raw`\eta \;=\; \frac{\mathrm{CumKL}^{\text{abl}}_{\text{method}}}{\mathrm{CumKL}^{\text{abl}}_{\text{oracle}}}\times 100\% \;\in\; [0,100]\%\,, \qquad \mathrm{RCK} \;=\; \frac{\mathrm{CumKL}_{\text{multi-action}}}{\mathrm{CumKL}^{\text{abl}}_{\text{oracle}}}\times 100\% \;\;(\text{may exceed } 100\%)`} />
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>III.1 · Headline</Kicker>
          <h2 className="sec">IOI: the agent beats random where it matters, and loses to EAP everywhere</h2>
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
            On Gemma the agent reaches 82.0% of an oracle that already knows every answer — +43.5%
            relative over random, and the exact paired permutation test says that is real
            (p = 0.031, the minimum attainable two-condition p at n = 5 — i.e. the agent beat
            random on <em>every single prompt</em>). It also edges the engineered bandit (74.4%)
            and plain UCB (73.7%). On Llama it does not: 52.1% with a CI touching random, behind
            the bandit. And on both models EAP&rsquo;s static ranking (91.9 / 95.2%) is untouched.
            The paper says this in the abstract, and the talk should too — then explain{' '}
            <em>why that was always the likely outcome</em>: EAP ranks by direct attribution to the
            logits computed on the very graph the candidates came from. It is close to an oracle
            for the pruned set. The agent&rsquo;s case never rested on out-ranking it (III.7).
          </p>
          <Fig caption={<>Bounded oracle efficiency across all four task×model cells, with bootstrap 95% CIs — rendered from <code>stats.json</code>. The two patterns to narrate: EAP is always first; the agent&rsquo;s rank degrades from Gemma to Llama, catastrophically so on multi-step.</>}>
            <EffBars />
            <Legend keys={['eap', 'ai_abl', 'bandit', 'ucb', 'greedy', 'random']} />
          </Fig>
          <Fig caption={<>Mean cumulative ablation KL per step (IOI). No method crosses the oracle envelope (dashed) — the bound is structural, all methods share the action. Gemma: the agent tracks EAP closely from step ~6. Llama: the agent sits with greedy/random from early on.</>}>
            <CumKLChart task="ioi" />
            <Legend keys={['oracle', 'eap', 'ai_abl', 'bandit', 'greedy']} />
          </Fig>
        </div>
      </section>

      <section className="band soft">
        <div className="reading">
          <Kicker>III.2 · The decomposition that keeps the paper honest</Kicker>
          <h2 className="sec">RCK: amplification is the action, not the intelligence</h2>
          <p>
            Free the agent to choose intervention <em>types</em> and its cumulative KL explodes past
            the ablation oracle — 1255% on Gemma IOI. It would have been easy to print that as the
            headline. The paper instead builds the control that dismantles it:{' '}
            <strong>action-matched baselines</strong>. Give the same steering action to random
            selection and you get 888%; to the bandit, 1185%. The bulk of the amplification is the
            steering lever itself. The full agent does top the Gemma ordering — EFE selection adds a
            real margin — but the paper&rsquo;s wording is exact: RCK “must not be read as a
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
          <p>
            The Llama row is the most instructive in the paper: the agent&rsquo;s RCK is{' '}
            <strong>91%</strong> — below the ablation oracle — because it mostly declined to steer
            (70/100 ablations). Same objective, same priors, different observation statistics,
            different emergent policy. A hard-coded “steer for the KL” heuristic cannot produce
            that reversal; a belief-driven policy can. This is the single best exhibit that the
            action selection is inference, not schedule.
          </p>
          <Fig caption={<>RCK on a log scale, agent vs action-matched baselines, IOI. Amber dashes: the ablation oracle at 100%. Gemma: everything with steering blows past 100 — the action dominates; the agent leads. Llama: the agent lands <em>below</em> 100 by choosing ablation.</>}>
            <RCKChart />
          </Fig>
          <Fig caption={<>Every action the agent took: 5 prompts × 20 steps per model, from the raw run logs. Gemma: one exploratory ablation opens each prompt (steps 1), then steering dominates (94/100; 1 patch). Llama: ablation-heavy throughout (70/100). Emergent, not scheduled.</>}>
            <ActionStrip />
          </Fig>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>III.3 · Steering</Kicker>
          <h2 className="sec">Dose-dependent control — and a selectivity null the paper keeps</h2>
          <p>
            The Golden-Gate-style test: 5 concept prompts, top-10 circuit-selected features each,
            multipliers m ∈ {'{'}0, 1.5, 2, 3, 5, 10{'}'} — against a matched random-feature
            control. Manipulability (H2a) is unambiguous: prediction changes rise monotonically
            with dose, 11/50 (Gemma) and 9/50 (Llama) at m = 10, binomial p &lt; 10⁻⁸ against
            chance. Mean KL of selected features at m = 10 is 4× the control&rsquo;s (0.367 vs
            0.086 on Gemma). And the steered outputs amplify <em>coherent concepts</em>: top-1
            probability mass holds (~0.43) while dominant tokens diversify toward
            geographic/landmark content rather than collapsing to function words.
          </p>
          <p>
            But selectivity (H2b) — are <em>circuit-selected</em> features more steerable than
            random active ones? — <strong>fails</strong>: 11 vs 8 and 9 vs 6 changes, Fisher exact
            n.s., at multipliers that are frankly off-distribution. The paper prints the null and
            the OOD caveat. For this audience the null is a feature: it is exactly the
            steering-vector reliability critique from the interpretability literature (large
            multipliers ≈ generic activation scaling), reproduced independently with a proper
            control.
          </p>
          <Fig caption={<>Dose-response from the raw steering JSONs: top-1 prediction changes out of 50, circuit-selected (teal) vs random-control (grey dashes). The gap at m = 10 is visible — and not significant. Honesty rendered as a chart.</>}>
            <SteeringChart />
          </Fig>
        </div>
      </section>

      <section className="band soft">
        <div className="reading">
          <Kicker>III.4 · The failure, diagnosed</Kicker>
          <h2 className="sec">Llama multi-step: 9.3%, and why</h2>
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
            Below random. The paper neither buries this nor waves at “architecture differences” —
            it runs the per-prompt diagnosis: the agent scores 4.4% and 15.2% on the two prompts
            whose oracle KL mass sits in <em>early layers</em> (oracle CumKL 0.461, 0.257), and
            recovers to 55.9% on the one late-layer prompt (0.016). The mechanism: with 16 layers
            cut into three role bins, functionally distinct early-middle layers collapse into one
            hidden state — the belief state <em>cannot represent</em> where the causal mass is. The
            targeted fix — six bins, one flag — quadruples efficiency to 37.8% without touching the
            EFE objective. Still behind the bandit; reported anyway.
          </p>
          <Callout title="The active-inference reading of the failure" tone="violet">
            This is the most model-diagnostic result in the paper: when the generative model&rsquo;s
            state space is too coarse for the environment, no objective on top of it can save you —
            and refining the state space, not the objective, is what helps. That is precisely the
            structure-learning moral this community would predict. The honest concession attached:
            the fix was chosen by a human post-hoc, not discovered by the agent. Structure learning
            remains future work (Part V, Q6).
          </Callout>
          <Fig caption={<>Multi-step cumulative KL. Gemma: agent respectable at 73%. Llama: the teal curve flatlines against an oracle whose mass it cannot see — the visual of a state-space failure.</>}>
            <CumKLChart task="multistep" />
            <Legend keys={['oracle', 'eap', 'ai_abl', 'bandit', 'greedy']} />
          </Fig>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>III.5 · Domains & dynamics</Kicker>
          <h2 className="sec">Task-dependent circuit geography, and a likelihood that actually learns</h2>
          <p>
            Across five cognitive domains × two models the agent is a reliable selector (77–94% on
            Gemma, 57–98% on Llama — above random everywhere, trading wins with bandit/greedy,
            below EAP). The finding with scientific content is the <strong>layer geography</strong>:
            factual domains (geography, history) put their top-10 causal mass in late layers;
            reasoning domains (logic, math) recruit early layers; science sits mixed — replicated
            across both architectures, and consistent with the ROME/factual-recall picture from the
            interpretability side. IOI&rsquo;s single highest-KL features sit at layers 24–25 of
            Gemma&rsquo;s 26 (e.g. <code>L25_P14_F4717</code>, KL 0.0015–0.013) — the late-layer
            name-mover signature, recovered at transcoder resolution.
          </p>
          <DomainTable />
          <h3 className="sub">The observation model stabilises</h3>
          <p>
            The Dirichlet-learned likelihood is not decoration: the mean per-step L1 drift of the
            KL-likelihood matrix halves over the budget (1.03×10⁻³ → 5.0×10⁻⁴ on Gemma IOI), and
            belief entropy falls monotonically. The agent ends each episode measurably less
            uncertain about how importance maps to observable KL than it began — online learning
            doing its one job.
          </p>
          <Fig caption={<>Mean L1 drift of the learned likelihood per update, Gemma IOI — from the run logs. Monotone decay, no oscillation: the Dirichlet posterior over A is converging.</>}>
            <AConvChart />
          </Fig>
          <Deep title="Deep dive: sensitivity — one knob matters, and it's not an agent knob">
            <p>
              Threshold sweep (Gemma IOI, baseline 82.0%): activation thresholds ±20% → 86.2 / 80.3%;
              KL thresholds +20% → 82.0%; KL thresholds <strong>−20% → 60.2%</strong>. Tightening the
              KL bins injects observation noise into every belief update — the discretisation is the
              sensitive interface between the continuous environment and the discrete agent.
              Meanwhile the agent&rsquo;s own hyperparameters are flat to the point of comedy: γ ∈
              {'{'}4, 16, 64{'}'}, η ∈ {'{'}0.5, 1, 2{'}'}, pragmatic weight ∈ {'{'}0.5, 1, 2{'}'} all
              leave Gemma IOI at 82.0% (Llama wobbles 49.1–54.5% with η only). Read as a pair, these
              two sweeps locate the model&rsquo;s real prior exactly where Part II said it was: in
              the discretisation, not in γ. The qualitative ordering (above random, competitive with
              bandit, below EAP) survives every configuration tested.
            </p>
          </Deep>
        </div>
      </section>

      <section className="band dark2">
        <div className="reading">
          <Kicker>III.6 · Scorecard</Kicker>
          <h2 className="sec">Four hypotheses, graded in public</h2>
          <div className="tbl-wrap">
            <table className="tbl">
              <caption>Pre-registered hypotheses vs outcomes (paper Table 10). H1/H3 reported Gemma / Llama.</caption>
              <thead>
                <tr><th>Hypothesis</th><th>Criterion</th><th>Result</th><th className="num">p</th><th>Verdict</th></tr>
              </thead>
              <tbody>
                <tr><td>H1 · Efficiency</td><td>POMDP-abl &gt; random, paired permutation</td><td>+43.5% / +7.8%</td><Num>0.031 / 0.41</Num><td>Accepted on Gemma; <strong>not on Llama</strong></td></tr>
                <tr><td>H2a · Manipulability</td><td>Steering changes &gt; chance</td><td>11/50 &amp; 9/50 at m=10</td><Num>&lt;10⁻⁸</Num><td>Accepted</td></tr>
                <tr><td>H2b · Selectivity</td><td>Selected &gt; control, Fisher exact</td><td>11 vs 8; 9 vs 6</td><Num>n.s.</Num><td><strong>Not supported</strong></td></tr>
                <tr><td>H3 · Discovery ≥ 50%</td><td>Bounded oracle efficiency</td><td>82.0 / 52.1 (IOI); 73.0 / 9.3 (multi-step)</td><Num>—</Num><td>Accepted except <strong>Llama multi-step</strong></td></tr>
                <tr><td>H4 · Transfer</td><td>Qualitative replication</td><td>Gemma full, Llama partial</td><Num>—</Num><td><strong>Partially</strong> accepted</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            Two rejections and two partials, printed in the paper&rsquo;s own hypothesis table. When
            the room probes the weak spots, the answer is on the slide already — which converts a
            gotcha into a discussion.
          </p>
        </div>
      </section>

      <section className="band">
        <div className="reading">
          <Kicker>III.7 · The objective summary</Kicker>
          <h2 className="sec">What survives</h2>
          <p>
            <strong>Established:</strong> EFE-guided selection significantly beats random on Gemma
            IOI with the strongest available test at n = 5; the agent is competitive with engineered
            heuristics on Gemma across IOI and all five domains; the exploration→exploitation and
            architecture-dependent action policies emerge from the generative model; the learned
            likelihood converges; transcoder features are dose-dependently causal; task-dependent
            layer geography replicates across two architectures. <strong>Not established:</strong>{' '}
            superiority over EAP anywhere; selectivity of circuit-selected features under steering;
            transfer of the selection advantage to the shallow model; multi-step competence under
            the 3-bin state space. <strong>The honest one-liner for the talk:</strong> on these
            benchmarks the value of active inference is not raw efficiency — a static attribution
            ranking wins that — it is a principled, adaptive, self-diagnosing <em>policy</em> for
            spending causal-verification budget, whose failures localise to interpretable model
            defects rather than mystery. Whether that trade is worth it is a fair fight, and Part V
            arms you for it.
          </p>
          <Callout title="Why EAP winning was structurally likely" tone="amber">
            The candidate set itself was drawn from the top of the attribution graph, and EAP ranks
            by direct-to-logit attribution <em>on that same graph</em>. The evaluation is selection
            efficiency <em>within</em> EAP&rsquo;s own shortlist — home turf. The agent starts from
            a κ-scaled version of those importances as a mere prior and must rediscover the ordering
            through 20 noisy, discretised observations. That EAP is hard to out-rank on ablation-KL
            in-distribution is the expected result; the agent&rsquo;s additions (uncertainty,
            multi-action, online learning) pay off precisely where a static list cannot adapt.
            Where that is worth the machinery — auditing under distribution shift, multi-prompt
            budgets, intervention types with different costs — is the future-work frontier, and a
            fair question for the room.
          </Callout>
          <div style={{ marginTop: '2.2rem' }}>
            <Link href="/demo" style={{ fontFamily: 'var(--grotesk)', fontWeight: 600, fontSize: '.9rem', textDecoration: 'none', background: 'var(--teal)', color: '#fff', padding: '11px 22px', borderRadius: 999 }}>
              Part IV: watch it run →
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
