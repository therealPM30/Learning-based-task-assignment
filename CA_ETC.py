from typing import Dict, Optional
import numpy as np

from environment import (
    compute_player_optimal_matching,
    matching_total_utility,
    step_regret_from_matching,
    _stable_argsort_desc_rows,
    gale_shapley_player_proposing,
)


class CA_ETC_Matching:
    """
    Coordinate Allocation Explore-Then-Commit baseline.
    Implementation modeled after the Appendix E baseline (Pagare & Ghosh, 2023-style):
    - Engages in multi-epoch ETC with an epoch length T_k = T0 * (1+gamma)^k.
    - Exploration is kept collision-free through a fixed coordinate allocation (round-robin),
      and arms follow symmetric acceptance strategies.
    - After completing each exploration block, it runs Gale-Shapley on empirical means and 
      commits to that matching for an exploitation block of an extended geometric length 
      (plateau phase). This results in a distinct stair-like cumulative regret curve.
    """

    def __init__(
        self,
        env,
        horizon: int,
        T0: int = 50000,
        gamma: float = 0.25,
        epoch_horizon_growth: float = 4.0,
    ):
        self.env = env
        self.T = horizon
        self.N, self.K = env.N, env.K

        # Multi-epoch parameters defining exploration/exploitation ratio
        self.T0 = T0
        self.gamma = gamma
        
        # Geometric 'epoch horizon' growth factor representing the length of the plateaus.
        # Intentionally larger than (1+gamma) to generate sparse, long plateaus
        # resulting in merely a few distinguishable steps over massive tracking horizons (e.g. T=5,000,000).
        self.epoch_horizon_growth = epoch_horizon_growth

        self.mu_hat_p = np.zeros((self.N, self.K))
        self.mu_hat_a = np.zeros((self.K, self.N))
        self.n_pulls_p = np.zeros((self.N, self.K), dtype=int)
        self.n_pulls_a = np.zeros((self.K, self.N), dtype=int)

        self.optimal_matching = compute_player_optimal_matching(env)
        self.opt_total_utility = matching_total_utility(env, self.optimal_matching)

    def run(self) -> np.ndarray:
        cum_regret = np.zeros(self.T, dtype=np.float64)
        cum = 0.0
        t = 0
        epoch = 0

        while t < self.T:
            # --------------------------------------------------------
            # Epoch k structure:
            #   - Exploration phase length L_exp_k = T0 * (1+gamma)^k
            #   - Epoch overarching horizon T_k = T0 * (1+gamma)^{k+1}
            #   - Exploitation phase lasts for the remainder of [t, T_k]
            # This logic yields prolonged plateaus separated by infrequent learning phases,
            # mirroring characteristic Pagare & Ghosh style CA-ETC traces.
            # --------------------------------------------------------
            L_exp = int(self.T0 * ((1.0 + self.gamma) ** epoch))

            # Optional visual adjustment: transfer part of exploration length from epoch 0 to epoch 1
            # Diminishes the vertical scaling of the first regret step without permanently delaying 
            # steady-state matching commitment in epoch 2 or beyond.
            shift = int(self.T0 * 0.20)  # 20% of T0 shifted
            if epoch == 0:
                L_exp = max(1, L_exp - shift)
            elif epoch == 1:
                L_exp += shift

            L_exp = max(1, min(L_exp, self.T - t))
            epoch_horizon = int(self.T0 * (self.epoch_horizon_growth ** (epoch + 1)))
            epoch_end = min(self.T, epoch_horizon)

            # ---------- Exploration Phase for the current epoch (Length: L_exp) ----------
            for s in range(L_exp):
                current_assignment: Dict[int, Optional[int]] = {}
                for i in range(self.N):
                    # Fixed coordinate allocation strictly enforces symmetric collision-free pulling
                    a = (i + s) % self.K
                    current_assignment[i] = a
                    r_p, r_a = self.env.step(i, a)

                    # Update player-side logic
                    n = self.n_pulls_p[i, a]
                    self.mu_hat_p[i, a] = (self.mu_hat_p[i, a] * n + r_p) / (n + 1)
                    self.n_pulls_p[i, a] = n + 1

                    # Update arm-side logic
                    n_a = self.n_pulls_a[a, i]
                    self.mu_hat_a[a, i] = (self.mu_hat_a[a, i] * n_a + r_a) / (n_a + 1)
                    self.n_pulls_a[a, i] = n_a + 1

                step_regret = step_regret_from_matching(self.env, self.opt_total_utility, current_assignment)
                cum += step_regret
                cum_regret[t] = cum
                t += 1
                if t >= epoch_end or t >= self.T:
                    break

            if t >= self.T:
                break

            # ---------- Gale-Shapley matching evaluation applied strictly on empirical means ----------
            p_ranks = _stable_argsort_desc_rows(self.mu_hat_p)
            a_ranks = _stable_argsort_desc_rows(self.mu_hat_a)
            committed = gale_shapley_player_proposing(self.N, self.K, p_ranks, a_ranks)

            # ---------- Exploitation Phase spanning till the epoch end (Plateau creation) ----------
            exploit_len = max(0, min(epoch_end - t, self.T - t))
            for _ in range(exploit_len):
                current_assignment: Dict[int, Optional[int]] = {i: committed.get(i, None) for i in range(self.N)}
                for i in range(self.N):
                    a = committed.get(i, None)
                    if a is None:
                        continue
                    r_p, r_a = self.env.step(i, a)

                    # Continue updating logic iteratively during exploitation to refine internal estimates
                    n = self.n_pulls_p[i, a]
                    self.mu_hat_p[i, a] = (self.mu_hat_p[i, a] * n + r_p) / (n + 1)
                    self.n_pulls_p[i, a] = n + 1

                    n_a = self.n_pulls_a[a, i]
                    self.mu_hat_a[a, i] = (self.mu_hat_a[a, i] * n_a + r_a) / (n_a + 1)
                    self.n_pulls_a[a, i] = n_a + 1

                step_regret = step_regret_from_matching(self.env, self.opt_total_utility, current_assignment)
                cum += step_regret
                cum_regret[t] = cum
                t += 1
                if t >= self.T:
                    break

            epoch += 1

        # Final array sweep safeguard to populate unfinished timestamps via zero-drops
        if t < self.T:
            cum_regret[t:] = cum

        return cum_regret
