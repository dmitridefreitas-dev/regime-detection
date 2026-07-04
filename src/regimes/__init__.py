"""regimes — market regime detection with a from-scratch Gaussian HMM.

Modules:
    hmm         2-state Gaussian hidden Markov model: Baum-Welch EM,
                filtered (causal) and smoothed (anticausal) probabilities,
                Viterbi decoding
    allocation  regime-aware exposure with honest lagged evaluation
    data        cached daily price downloads
"""

from regimes.hmm import GaussianHMM

__all__ = ["GaussianHMM"]
