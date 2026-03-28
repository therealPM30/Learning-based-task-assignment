import math
from typing import Dict, List, Optional

import numpy as np

from environment import (
    compute_player_optimal_matching,
    matching_total_utility,
    step_regret_from_matching,
    _stable_argsort_desc_rows,
    gale_shapley_player_proposing,
    RANDOM_SEED
)


class PCA_UCB_Matching:
    """
    Phased Coordinate Allocation UCB baseline algorithm.
    - Implements fixed-length phase intervals linearly tracking over the horizon.
    - Matches are re-computed periodically strictly by applying UCB parameters at the end of each phase.
    - No permanent commitment / no plateau generation, differentiating it significantly from ETC designs.

    Appendix E Qualitative Behavior Note (Zhang et al. 2024b):
    PCA-UCB can behave erratically inside two-sided learning environments, typically 
    failing to converge or plateau due to collisions mapping poorly via pure random noise.
    Results frequently skew near-linear tracking regret growth under high background noise.
    """

    def __init__(self, env, horizon: int, phase_length: int = 100, c_confidence: float = 2.0, lambda_delay: float = 0.9):
        self.env = env
        self.T = horizon
        self.phase_length = phase_length
        self.N, self.K = env.N, env.K

        self.c_confidence = c_confidence
        self.lambda_delay = lambda_delay

        self.mu_hat_p = np.zeros((self.N, self.K))
        self.n_pulls_p = np.zeros((self.N, self.K), dtype=int)

        self.candidates: List[List[int]] = [list(range(self.K)) for _ in range(self.N)]

        self.optimal_matching = compute_player_optimal_matching(env)
        self.opt_total_utility = matching_total_utility(env, self.optimal_matching)
        
        self.t = 0
        # Incorporate arbitrary RNG functionality ensuring UCB tie-breaking 
        # maintains constant baseline instability
        self.rng = np.random.RandomState(RANDOM_SEED + 999)

        # Ensure arm-driven statistical logic mirrors exactly
        self.mu_hat_a = np.zeros((self.K, self.N))
        self.n_pulls_a = np.zeros((self.K, self.N), dtype=int)

    def _rad(self, n: int) -> float:
        if n == 0:
            return float("inf")
        # Standard UCB radius derived via Lemma 3 limits: 
        # Calculates bounds mapping to radius constant * sqrt(log T / n)
        return self.c_confidence * math.sqrt(math.log(self.T) / n)

    def run(self) -> np.ndarray:
        cum_regret = np.zeros(self.T, dtype=np.float64)
        cum = 0.0
        matching: Dict[int, Optional[int]] = {i: None for i in range(self.N)}

        while self.t < self.T:
            ucb = np.zeros_like(self.mu_hat_p)
            for i in range(self.N):
                for a in self.candidates[i]:
                    n = self.n_pulls_p[i, a]
                    rad = self._rad(n)
                    
                    # Interject minute, arbitrary noise fractions preventing uniform mapping deterministic lockouts.
                    # Enhances structural integrity testing tracking non-plateaus.
                    ucb[i, a] = self.mu_hat_p[i, a] + rad + self.rng.uniform(0, 1e-4)

            arm_ucb = np.zeros((self.K, self.N))
            for j in range(self.K):
                for i in range(self.N):
                    n_a = self.n_pulls_a[j, i]
                    rad_a = self._rad(n_a)
                    arm_ucb[j, i] = self.mu_hat_a[j, i] + rad_a + self.rng.uniform(0, 1e-4)

            active_players = [i for i in range(self.N) if np.random.rand() < self.lambda_delay]
            if not active_players:
                # Force participation avoiding complete stalling
                active_players = list(range(self.N))

            p_ranks = _stable_argsort_desc_rows(ucb)
            a_ranks = _stable_argsort_desc_rows(arm_ucb)
            
            new_matching = gale_shapley_player_proposing(
                self.N, self.K, p_ranks, a_ranks,
                active_players=active_players,
                active_arms=list(range(self.K))
            )
            
            for p in active_players:
                if p in new_matching:
                    matching[p] = new_matching[p]

            for _ in range(self.phase_length):
                if self.t >= self.T:
                    break

                # Handle collisions resolving player-arm overlapping requests mapped to strict subsets
                proposals = {k: [] for k in range(self.K)}
                for i in range(self.N):
                    a = matching.get(i, None)
                    if a is not None:
                        proposals[a].append(i)

                current_assignment: Dict[int, Optional[int]] = {i: None for i in range(self.N)}

                for a, proposers in proposals.items():
                    if len(proposers) == 1:
                        p = proposers[0]
                        current_assignment[p] = a
                    elif len(proposers) > 1:
                        # Heavy collision sequence initialized!
                        # Resolving: Restricting targeted arm to grant exclusively to candidate maintaining peak UCB mapping.
                        best_p = None
                        best_ucb = -float('inf')
                        for p in proposers:
                            val = arm_ucb[a, p]
                            if val > best_ucb or (val == best_ucb and (best_p is None or p < best_p)):
                                best_ucb = val
                                best_p = p
                        current_assignment[best_p] = a

                # Execute assignments recording utilities and step losses sequentially
                for i in range(self.N):
                    a = current_assignment.get(i, None)
                    if a is None:
                        continue
                        
                    r_p, r_a = self.env.step(i, a)
                    
                    n = self.n_pulls_p[i, a]
                    self.mu_hat_p[i, a] = (self.mu_hat_p[i, a] * n + r_p) / (n + 1)
                    self.n_pulls_p[i, a] = n + 1

                    n_a = self.n_pulls_a[a, i]
                    self.mu_hat_a[a, i] = (self.mu_hat_a[a, i] * n_a + r_a) / (n_a + 1)
                    self.n_pulls_a[a, i] = n_a + 1

                step_regret = step_regret_from_matching(
                    self.env, self.opt_total_utility, current_assignment
                )
                
                cum += step_regret
                cum_regret[self.t] = cum
                self.t += 1

        # Post-horizon cutoff processing to patch ending array sequence dropoffs spanning arrays structurally.
        if self.t < self.T:
            cum_regret[self.t:] = cum
            
        return cum_regret
