# Code walkthrough — how every module actually works

Companion to `how-it-works.md` (concepts). This is the code-level defense.

## Map

| File | Role | Key entry points |
|---|---|---|
| `hmm.py` | the from-scratch HMM | `GaussianHMM.fit`, `.filtered_probabilities`, `.smoothed_probabilities`, `.viterbi` |
| `allocation.py` | exposure rule + honest evaluation | `exposure_from_prob`, `evaluate`, `summarize` |
| `data.py` | cached price downloads | `load_prices` |

## `hmm.py` — the model in ~200 lines

### Initialisation (deterministic on purpose)

Split observations at the median absolute deviation: the quiet half seeds state 0's
mean/std, the noisy half state 1's. Transition starts at [[.95,.05],[.05,.95]],
initial at [.5,.5]. No RNG anywhere → every fit is byte-identical
(`test_deterministic_fit`). K-means or random restarts would be the usual
alternatives; for a 2-state vol-separated problem the median split lands in the
right basin every time.

### `_forward` — filtering and the likelihood, one pass

```python
step      = initial * emission[0]          # or (alpha[t-1] @ transition) * emission[t]
scales[t] = step.sum()
alpha[t]  = step / scales[t]
```

Three facts to be able to state:

1. **The normalised alpha IS the filtered posterior** P(state_t | x_1..t). The
   unnormalised forward variable is P(state_t, x_1..t); dividing by its sum
   conditions on the observations.
2. **The scales ARE the likelihood**: scales[t] = P(x_t | x_1..t−1), so
   `log P(x) = Σ log(scales)` — the number EM monitors.
3. **Scaling is the underflow fix**: raw forward products shrink like
   (typical density)^n and underflow long before 8,000 observations. Normalising
   each step keeps everything O(1) while preserving ratios. (Log-space + logsumexp
   is the equally valid alternative; scaling keeps the textbook algebra.)

### `_backward` — the mirror pass

beta[t] ∝ P(x_{t+1}..x_n | state_t), computed backwards and divided by the *shared*
forward scales so that `alpha * beta` is correctly normalised when combined.

### The E-step quantities

- `gamma = alpha * beta`, row-normalised → P(state_t | ALL data) (the smoothed
  posterior).
- `xi[t,i,j] ∝ alpha[t,i] · A[i,j] · emission[t+1,j] · beta[t+1,j]` → the expected
  transition counts. In code it's one broadcast:
  `alpha[:-1,:,None] * transition[None,:,:] * (emission[1:,None,:] * beta[1:,None,:])`,
  normalised per time step. Shapes: (n−1, 2, 2).

### The M-step (all responsibility-weighted averages)

- `initial = gamma[0]`
- `transition[i,j] = Σ_t xi[t,i,j] / Σ_t gamma[t,i]` (expected transitions out of i,
  normalised), then row-renormalised for float safety.
- `means = Σ_t w_t·x_t`, `vars = Σ_t w_t·(x_t − mean)²` with `w = gamma/Σgamma`.
- **Sigma floor** (1e-8): stops a state collapsing onto a single point (a known EM
  degeneracy where likelihood → ∞).

Convergence: stop when the loglik gain < tol; the per-iteration histories are kept
and a test asserts they never decrease (EM's defining guarantee).

### Identifiability

After fitting, if state 0 has the larger std, swap everything (means, stds, initial,
and `transition[np.ix_(order, order)]` — both rows *and* columns). Mixtures are
only defined up to label permutation; pinning "state 0 = calm" makes downstream code
and plots stable.

### `viterbi` — hard decoding

Log-space DP: `score[t,j] = max_i(score[t−1,i] + logA[i,j]) + log emission[t,j]`
with backpointers, then a backward trace. The `+1e-300` inside the logs guards
log(0) for impossible transitions. Max-sum (best single path) vs the forward
algorithm's sum-product (marginals) — and like smoothing, Viterbi is anticausal:
fine for description, lookahead if traded.

### Small utilities

`stationary_distribution` = the eigenvector of Aᵀ for eigenvalue 1, normalised;
`expected_duration_days` = 1/(1 − p_stay) — geometric sojourn time, the "storms
last ~33 days" number.

## `allocation.py` — the honest evaluation shim

`exposure_from_prob` clips P(calm) into [floor, cap]. `evaluate` reproduces the
honest-backtester engine contract in miniature — `positions.shift(lag)` with lag ≥ 1
enforced (same-bar fills raise), turnover charged at `cost_bps`, first-position
entry charged via `fillna(held.abs())`. `summarize` = geometric annual return, vol,
Sharpe (ddof=1, rf=0), max drawdown off the running peak.

## The causal loop in the notebook (worth reciting)

For each year Y ≥ 2003: fit on returns with `index.year < Y` (expanding window);
compute **filtered** probabilities on data through Y using those past-fitted
parameters; keep only year Y's values; concatenate. Every number at time t therefore
depends on (a) parameters from before Y and (b) observations up to t. The two
cheating variants differ in exactly one ingredient each — full-sample parameters, or
smoothed probabilities — which is what lets the ladder attribute the Sharpe gap.

## The tests, as a defense layer

- **Parameter recovery**: simulate a chain with known means/stds/transition
  (0.7%/2.0% daily vols, 0.98/0.96 stay-probs, 6,000 obs); the fit must land within
  10% on vols, 0.02–0.03 on transition diagonals.
- **EM monotonicity**: diffs of the loglik history ≥ −1e-7.
- **Prefix property on filtered probs** (the trading-critical one): filtered on
  x[:k] equals the first k rows of filtered on all of x, at three cutoffs, to 1e-12.
- **Smoothed is anticausal**: the same comparison *must* differ (> 1e-3) — the
  contrast pair is the whole point of the repo.
- **Viterbi block recovery**: planted calm/wild/calm blocks come back >90% correct.
- **Allocation**: same-bar refusal; hand-computed cost example.

## Grilling Q&A (implementation level)

- *Why does normalising alpha give the filtered posterior?* Bayes: unnormalised
  forward = prior-propagated belief × likelihood of today's observation; the
  normaliser is the observation's marginal. It's a recursive Bayes filter — the same
  structure as a Kalman filter with discrete states.
- *Why can't you just fit two Gaussians (a mixture) without the chain?* You'd lose
  persistence — the transition matrix is what makes today's regime informative about
  tomorrow's, which is the entire tradeable content (stay-probs of 0.97–0.99 ⇒
  regimes are forecastable at 1-day horizon).
- *EM found a local optimum — how do you know it's the right one?* Deterministic
  init in the vol-separated basin, parameter-recovery tests on synthetic data, and
  the fitted states matching known market episodes. For production: multiple
  restarts and BIC across state counts (named next step).
- *Why filtered and not one-step-ahead predicted probabilities?* Filtered at t is
  knowable at t's close and the engine lags execution by a bar anyway — so the
  position for t+1 uses information through t. Predicted probs (alpha @ A) would be
  marginally more conservative; with stay-probs ~0.98 the difference is tiny.
- *Where does the Sharpe 1.74 fantasy physically come from?* The backward pass:
  beta at time t contains the crash at t+k, so smoothed P(turbulent) rises *before*
  the crash in calendar time. The strategy de-risks on information that did not
  exist yet.
