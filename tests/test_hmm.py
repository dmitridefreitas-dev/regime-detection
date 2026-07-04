"""The HMM must recover known parameters, keep probabilities honest,
and — critically — filtered probabilities must be causal."""

import numpy as np
import pytest

from regimes import GaussianHMM


def simulate(n=6000, seed=3):
    """Ground-truth 2-state chain: calm (low vol) and turbulent (high vol)."""
    rng = np.random.default_rng(seed)
    true = {
        "means": np.array([0.0008, -0.0005]),
        "stds": np.array([0.007, 0.020]),
        "transition": np.array([[0.98, 0.02], [0.04, 0.96]]),
    }
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = true["transition"][states[t - 1], states[t - 1]]
        states[t] = states[t - 1] if rng.random() < stay else 1 - states[t - 1]
    x = rng.normal(true["means"][states], true["stds"][states])
    return x, states, true


class TestParameterRecovery:
    def test_recovers_generating_parameters(self):
        x, _, true = simulate()
        model = GaussianHMM.fit(x)

        assert model.stds[0] == pytest.approx(true["stds"][0], rel=0.10)
        assert model.stds[1] == pytest.approx(true["stds"][1], rel=0.10)
        assert model.means[0] == pytest.approx(true["means"][0], abs=5e-4)
        assert model.means[1] == pytest.approx(true["means"][1], abs=3e-3)
        assert model.transition[0, 0] == pytest.approx(true["transition"][0, 0], abs=0.02)
        assert model.transition[1, 1] == pytest.approx(true["transition"][1, 1], abs=0.03)

    def test_em_loglik_is_monotone(self):
        x, _, _ = simulate(n=2000, seed=5)
        model = GaussianHMM.fit(x)
        diffs = np.diff(model.em_history)
        assert (diffs >= -1e-7).all()  # allow float noise, forbid real decreases

    def test_state_zero_is_calm_by_convention(self):
        x, _, _ = simulate(seed=8)
        model = GaussianHMM.fit(x)
        assert model.stds[0] < model.stds[1]

    def test_deterministic_fit(self):
        x, _, _ = simulate(n=2000, seed=9)
        a, b = GaussianHMM.fit(x), GaussianHMM.fit(x)
        np.testing.assert_array_equal(a.means, b.means)
        np.testing.assert_array_equal(a.transition, b.transition)


class TestProbabilities:
    def test_rows_sum_to_one(self):
        x, _, _ = simulate(n=2000, seed=11)
        model = GaussianHMM.fit(x)
        np.testing.assert_allclose(model.filtered_probabilities(x).sum(axis=1), 1.0)
        np.testing.assert_allclose(model.smoothed_probabilities(x).sum(axis=1), 1.0)

    def test_filtered_is_causal_prefix_property(self):
        # THE test that matters for trading use: filtered probs at time t
        # must not change when future observations are appended.
        x, _, _ = simulate(n=1500, seed=13)
        model = GaussianHMM.fit(x)
        full = model.filtered_probabilities(x)
        for cutoff in (300, 800, 1400):
            prefix = model.filtered_probabilities(x[:cutoff])
            np.testing.assert_allclose(prefix, full[:cutoff], atol=1e-12)

    def test_smoothed_uses_the_future(self):
        # The contrast: smoothing at time t DOES change with future data.
        x, _, _ = simulate(n=1500, seed=13)
        model = GaussianHMM.fit(x)
        full = model.smoothed_probabilities(x)[:800]
        prefix = model.smoothed_probabilities(x[:800])
        assert np.abs(full - prefix).max() > 1e-3

    def test_smoothed_beats_filtered_at_state_recovery(self):
        # Hindsight should classify at least as well as real time.
        x, states, _ = simulate(seed=17)
        model = GaussianHMM.fit(x)
        filtered_acc = ((model.filtered_probabilities(x)[:, 1] > 0.5) == states).mean()
        smoothed_acc = ((model.smoothed_probabilities(x)[:, 1] > 0.5) == states).mean()
        assert smoothed_acc >= filtered_acc
        assert smoothed_acc > 0.85


class TestViterbi:
    def test_recovers_block_structure(self):
        rng = np.random.default_rng(21)
        calm = rng.normal(0.0005, 0.006, 1000)
        wild = rng.normal(-0.001, 0.025, 400)
        x = np.concatenate([calm, wild, calm])
        model = GaussianHMM.fit(x)
        path = model.viterbi(x)
        assert path[:1000].mean() < 0.1      # mostly calm
        assert path[1000:1400].mean() > 0.9  # mostly turbulent
        assert path[1400:].mean() < 0.1

    def test_stationary_distribution_sums_to_one(self):
        x, _, _ = simulate(n=2000, seed=23)
        model = GaussianHMM.fit(x)
        assert model.stationary_distribution().sum() == pytest.approx(1.0)
        assert (model.expected_duration_days() > 1.0).all()


class TestValidation:
    def test_too_short_rejected(self):
        with pytest.raises(ValueError, match="at least 10"):
            GaussianHMM.fit(np.array([0.01, -0.01]))
