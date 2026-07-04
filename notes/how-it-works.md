# How this project works — study notes

Plain-English walkthrough. Read alongside `notebooks/study.ipynb`.

## What this project is

A hidden Markov model for market regimes, built from scratch, plus the honest answer
to "can you trade it?". The two deliverables: (1) the HMM implementation itself —
Baum-Welch EM, filtered/smoothed inference, Viterbi — validated by parameter recovery
on simulated data, and (2) the *lookahead ladder*, which measures exactly how much
Sharpe each kind of cheating buys in a regime strategy.

## The model, in plain terms

Assume each day the market sits in one of two hidden states — calm or turbulent — and
today's return is a draw from that state's Gaussian. The state follows a Markov chain:
tomorrow's regime depends only on today's, with high persistence. You never observe
the state; you observe returns and must infer it. Fit to SPY: calm ≈ 11% vol with
+24% drift, ~69% of days, spells lasting months; turbulent ≈ 29% vol with −15% drift,
arriving in clusters (2000–03, 2008–09, 2020, 2022), spells lasting weeks.

## The algorithms (what to be able to explain)

- **Forward pass / filtering.** Recursively update P(state | data so far): propagate
  yesterday's belief through the transition matrix, multiply by today's observation
  likelihood, normalise. The normalising constants are saved — their logs sum to the
  log-likelihood, and *scaling* this way (rather than working in log space) avoids
  numerical underflow over 8,000+ days while keeping the classic algebra.
- **Backward pass / smoothing.** A mirror recursion from the end of the sample;
  combining forward and backward gives P(state | ALL data). Strictly better
  classification — strictly illegal to trade.
- **Baum-Welch (EM).** E-step: compute state responsibilities (gamma) and transition
  responsibilities (xi) from forward-backward. M-step: re-estimate means, vols,
  transition matrix as responsibility-weighted averages. Each iteration provably
  never decreases the likelihood (a test asserts monotonicity).
- **Viterbi.** Dynamic programming for the single most likely state *path* (max-sum
  instead of sum-product). Used for description; it is also anticausal.
- **Identifiability details:** states are only defined up to relabelling, so state 0
  is forced to be the lower-vol one after fitting; initialisation is a deterministic
  median split on |deviation|, so every fit is exactly reproducible with no seed.

## The study and its two results

**Result 1 — regime-scaling transforms risk rather than creating return.** Exposure =
filtered P(calm), parameters refit each January on past data only, one-bar lag, 2 bps
costs, evaluated 2003–2026: Sharpe 0.78 vs buy-and-hold's 0.67, max drawdown −17% vs
−55%, annual return 7.1% vs 11.4%. You pay ~4 points of return for half the vol. Who
should take that trade: anyone drawdown- or leverage-constrained. Who shouldn't: a
pure return maximiser.

**Result 2 — the lookahead ladder.** Same rule, three information sets:

| variant | Sharpe | what it cheats on |
|---|---|---|
| causal | 0.78 | nothing |
| param lookahead | 0.77 | parameters fit on the full sample |
| smoothed | 1.74 | probabilities that saw the future |

Parameter lookahead is nearly free because regime parameters are stable across
decades — annual refitting was correct hygiene but not where danger lived. Probability
lookahead more than doubles Sharpe: smoothing starts exiting *before* crashes because
the backward pass has already seen them. The practical warning: HMM libraries hand
back smoothed probabilities (or Viterbi paths) as the default output, so a naive
"regime strategy backtest" very often reports the 1.74-style number.

## How this connects to the other repos

The prefix-property test on filtered probabilities is the same causality contract as
honest-backtester's signal tests; `allocation.evaluate` reproduces that repo's
execution rules (lag ≥ 1 enforced, costs on turnover) in miniature. The portfolio's
single recurring theme: identify precisely what was knowable when, and let nothing
else into the result.

## Likely interview questions

- *Why scale rather than work in log space?* Both fix underflow; scaling preserves the
  textbook recursions and gives the log-likelihood for free as the sum of log
  normalisers. Log-space needs logsumexp at each step — equally valid, more bookkeeping.
- *Why is EM guaranteed to increase likelihood?* Standard EM argument: the E-step
  builds a tight lower bound (expected complete-data log-likelihood), the M-step
  maximises it; Jensen's inequality does the rest. The test asserts the monotonicity.
- *Why did you refit annually rather than daily?* The ladder shows parameter
  lookahead is worth ~0.01 of Sharpe here — refit frequency is second-order. Filtering
  (the part that matters) updates daily regardless.
- *Isn't this just volatility targeting?* Largely, yes — that is stated in the
  limitations, and running vol targeting as the fair benchmark is the named next
  experiment. The HMM adds an explicit persistence model and a drift difference, but
  the vol separation does most of the work.
- *Why two states?* Parsimony and interpretability for a first pass; BIC across state
  counts is the principled selection, and a third state usually splits turbulence
  into correction vs crisis.
