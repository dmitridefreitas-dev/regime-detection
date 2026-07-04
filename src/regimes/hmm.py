"""Two-state Gaussian hidden Markov model, implemented from scratch.

The model: an unobserved Markov chain switches between two states (calm /
turbulent), and each day's return is drawn from a Gaussian whose mean and
volatility depend on the current state. Everything is estimated by
Baum-Welch (EM with the scaled forward-backward recursions — scaling, not
log-space, keeps the classic algebra and avoids underflow over decades of
daily data).

The distinction this module is built to make obvious:

    filtered_probabilities(x)[t] = P(state_t | x_1..x_t)   — causal,
        computable in real time, the only version a trading rule may use
    smoothed_probabilities(x)[t] = P(state_t | x_1..x_N)   — uses the
        future; beautiful for historical narrative, lookahead if traded

States are relabelled after fitting so index 0 is the lower-volatility
("calm") state — Gaussian mixtures are only identified up to permutation.
Initialisation is deterministic (split observations at the median absolute
deviation), so fits are exactly reproducible without seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_SIGMA_FLOOR = 1e-8


def _gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class GaussianHMM:
    """A fitted 2-state Gaussian HMM. Build with `GaussianHMM.fit(x)`."""

    means: np.ndarray                 # (2,) per-state observation mean
    stds: np.ndarray                  # (2,) per-state observation std
    transition: np.ndarray            # (2,2), rows sum to 1
    initial: np.ndarray               # (2,) initial state distribution
    log_likelihood: float
    em_history: np.ndarray = field(repr=False)   # loglik per EM iteration

    # ------------------------------------------------------------- fitting

    @classmethod
    def fit(cls, x, n_iter: int = 300, tol: float = 1e-9) -> "GaussianHMM":
        """Baum-Welch EM. Deterministic initialisation, monotone loglik."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) < 10:
            raise ValueError("x must be a 1-D array with at least 10 observations")

        # Deterministic start: quiet half vs noisy half by |deviation|.
        deviation = np.abs(x - np.median(x))
        quiet = deviation <= np.median(deviation)
        means = np.array([x[quiet].mean(), x[~quiet].mean()])
        stds = np.maximum(np.array([x[quiet].std(), x[~quiet].std()]), _SIGMA_FLOOR)
        transition = np.array([[0.95, 0.05], [0.05, 0.95]])
        initial = np.array([0.5, 0.5])

        history = []
        previous = -np.inf
        for _ in range(n_iter):
            alpha, scales = _forward(x, means, stds, transition, initial)
            loglik = float(np.log(scales).sum())
            history.append(loglik)
            if loglik - previous < tol:
                break
            previous = loglik

            beta = _backward(x, means, stds, transition, scales)
            emission = np.stack(
                [_gaussian_pdf(x, means[i], stds[i]) for i in (0, 1)], axis=1
            )

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            # xi[t, i, j] ~ P(state_t = i, state_{t+1} = j | data)
            xi = (
                alpha[:-1, :, None]
                * transition[None, :, :]
                * (emission[1:, None, :] * beta[1:, None, :])
            )
            xi /= xi.sum(axis=(1, 2), keepdims=True)

            initial = gamma[0]
            transition = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
            transition /= transition.sum(axis=1, keepdims=True)
            weights = gamma / gamma.sum(axis=0, keepdims=True)
            means = (weights * x[:, None]).sum(axis=0)
            variances = (weights * (x[:, None] - means) ** 2).sum(axis=0)
            stds = np.maximum(np.sqrt(variances), _SIGMA_FLOOR)

        # Identifiability: state 0 = calm (lower vol).
        if stds[0] > stds[1]:
            order = [1, 0]
            means, stds = means[order], stds[order]
            initial = initial[order]
            transition = transition[np.ix_(order, order)]

        return cls(
            means=means, stds=stds, transition=transition, initial=initial,
            log_likelihood=history[-1], em_history=np.array(history),
        )

    # ---------------------------------------------------------- inference

    def filtered_probabilities(self, x) -> np.ndarray:
        """P(state_t | observations up to and including t). Causal."""
        x = np.asarray(x, dtype=float)
        alpha, _ = _forward(x, self.means, self.stds, self.transition, self.initial)
        return alpha

    def smoothed_probabilities(self, x) -> np.ndarray:
        """P(state_t | ALL observations). Uses the future — never trade on it."""
        x = np.asarray(x, dtype=float)
        alpha, scales = _forward(x, self.means, self.stds, self.transition, self.initial)
        beta = _backward(x, self.means, self.stds, self.transition, scales)
        gamma = alpha * beta
        return gamma / gamma.sum(axis=1, keepdims=True)

    def viterbi(self, x) -> np.ndarray:
        """Most likely state path (hard assignments), in log space."""
        x = np.asarray(x, dtype=float)
        log_emission = np.stack(
            [np.log(_gaussian_pdf(x, self.means[i], self.stds[i]) + 1e-300)
             for i in (0, 1)], axis=1,
        )
        log_transition = np.log(self.transition + 1e-300)

        n = len(x)
        score = np.zeros((n, 2))
        backpointer = np.zeros((n, 2), dtype=int)
        score[0] = np.log(self.initial + 1e-300) + log_emission[0]
        for t in range(1, n):
            candidates = score[t - 1][:, None] + log_transition
            backpointer[t] = candidates.argmax(axis=0)
            score[t] = candidates.max(axis=0) + log_emission[t]

        states = np.zeros(n, dtype=int)
        states[-1] = score[-1].argmax()
        for t in range(n - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]
        return states

    def stationary_distribution(self) -> np.ndarray:
        """Long-run fraction of time in each state."""
        eigenvalues, eigenvectors = np.linalg.eig(self.transition.T)
        stationary = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
        return stationary / stationary.sum()

    def expected_duration_days(self) -> np.ndarray:
        """Mean sojourn length per state: 1 / (1 - p_stay)."""
        return 1.0 / (1.0 - np.diag(self.transition))


# ---------------------------------------------------------------- recursions

def _forward(x, means, stds, transition, initial):
    """Scaled forward pass. Returns (filtered probabilities, scale factors)."""
    n = len(x)
    emission = np.stack([_gaussian_pdf(x, means[i], stds[i]) for i in (0, 1)], axis=1)
    alpha = np.zeros((n, 2))
    scales = np.zeros(n)

    step = initial * emission[0]
    scales[0] = step.sum()
    alpha[0] = step / scales[0]
    for t in range(1, n):
        step = (alpha[t - 1] @ transition) * emission[t]
        scales[t] = step.sum()
        alpha[t] = step / scales[t]
    return alpha, scales


def _backward(x, means, stds, transition, scales):
    """Scaled backward pass, sharing the forward scale factors."""
    n = len(x)
    emission = np.stack([_gaussian_pdf(x, means[i], stds[i]) for i in (0, 1)], axis=1)
    beta = np.zeros((n, 2))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (transition @ (emission[t + 1] * beta[t + 1])) / scales[t + 1]
    return beta
