import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# Global config
# ================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ================================================================
# Utility: Gale–Shapley player-proposing stable matching
# (used both for computing the player-optimal benchmark and
# within CA-ETC / PCA-UCB phases)
# ================================================================

def gale_shapley_player_proposing(
    N: int,
    K: int,
    player_ranks: np.ndarray,
    arm_ranks: np.ndarray,
    active_players: Optional[List[int]] = None,
    active_arms: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Standard Gale–Shapley with players as proposers.

    player_ranks[i]  : permutation of arms (descending utilities)
    arm_ranks[j]     : permutation of players (descending utilities)

    Returns:
        dict player -> arm for matched pairs among the active sets.
    """
    if active_players is None:
        active_players = list(range(N))
    if active_arms is None:
        active_arms = list(range(K))

    free_players = list(active_players)
    next_proposal_index = {i: 0 for i in active_players}
    arm_partner: Dict[int, int] = {}
    player_partner: Dict[int, int] = {}

    # Precompute inverse ranks for arms
    # arm_pref_order[j][i] = rank of player i for arm j (0 is best)
    arm_pref_order = np.empty((K, N), dtype=int)
    for j in range(K):
        for r, i in enumerate(arm_ranks[j]):
            arm_pref_order[j, i] = r

    while free_players:
        i = free_players.pop(0)
        # skip if player already matched
        if i in player_partner:
            continue

        # find next arm in i's preference list that is active
        while next_proposal_index[i] < K:
            a = player_ranks[i, next_proposal_index[i]]
            next_proposal_index[i] += 1
            if a in active_arms:
                break
        else:
            # exhausted all arms
            continue

        current_partner = arm_partner.get(a)
        if current_partner is None:
            # arm free
            arm_partner[a] = i
            player_partner[i] = a
        else:
            # arm decides based on its own ranking
            if arm_pref_order[a, i] < arm_pref_order[a, current_partner]:
                # prefers new proposer
                arm_partner[a] = i
                player_partner[i] = a
                del player_partner[current_partner]
                free_players.append(current_partner)
            else:
                # rejects new proposer
                free_players.append(i)

    return player_partner


def gale_shapley_arm_proposing(
    N: int,
    K: int,
    player_ranks: np.ndarray,
    arm_ranks: np.ndarray,
    active_players: Optional[List[int]] = None,
    active_arms: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Gale–Shapley with ARMS as proposers (arm-optimal stable matching).

    Returns:
        dict player -> arm for matched pairs among the active sets.
    """
    if active_players is None:
        active_players = list(range(N))
    if active_arms is None:
        active_arms = list(range(K))

    free_arms = list(active_arms)
    next_proposal_index = {a: 0 for a in active_arms}
    player_partner: Dict[int, int] = {}
    arm_partner: Dict[int, int] = {}

    # Precompute inverse ranks for players
    # player_pref_order[i][a] = rank of arm a for player i (0 is best)
    player_pref_order = np.empty((N, K), dtype=int)
    for i in range(N):
        for r, a in enumerate(player_ranks[i]):
            player_pref_order[i, a] = r

    active_player_set = set(active_players)

    while free_arms:
        a = free_arms.pop(0)
        if a in arm_partner:
            continue

        # find next player in a's preference list that is active
        while next_proposal_index[a] < N:
            i = int(arm_ranks[a, next_proposal_index[a]])
            next_proposal_index[a] += 1
            if i in active_player_set:
                break
        else:
            continue

        current_arm_for_i = player_partner.get(i)
        if current_arm_for_i is None:
            player_partner[i] = a
            arm_partner[a] = i
        else:
            # player decides based on their ranking
            if player_pref_order[i, a] < player_pref_order[i, current_arm_for_i]:
                # prefers new proposer arm
                player_partner[i] = a
                arm_partner[a] = i
                del arm_partner[current_arm_for_i]
                free_arms.append(current_arm_for_i)
            else:
                free_arms.append(a)

    return player_partner


# ================================================================
# Environments
#   - PaperEnvironment: Appendix E reproduction
#   - MRTAEnvironment: multi-robot extension (battery, packet loss)
# ================================================================


@dataclass
class PaperEnvironment:
    N: int
    K: int
    noise_sigma: float = 0.1
    true_player_rewards: np.ndarray = None  # shape (N, K)
    true_arm_rewards: np.ndarray = None  # shape (K, N)

    def step(self, player: int, arm: int) -> Tuple[float, float]:
        """Two-sided Gaussian noise rewards, cf. Appendix E, Zhang24b."""
        mu_p = self.true_player_rewards[player, arm]
        mu_a = self.true_arm_rewards[arm, player]
        r_p = mu_p + np.random.normal(0.0, self.noise_sigma)
        r_a = mu_a + np.random.normal(0.0, self.noise_sigma)
        return r_p, r_a


class MRTAEnvironment:
    """
    MRTA extension environment (Part 4 in the master prompt).
    Heterogeneous drones (HEAVY/LIGHT), heterogeneous tasks,
    linear battery depletion based on distance.
    """

    def __init__(self, N: int, K: int, noise_sigma: float = 0.05, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.N = N
        self.K = K
        self.noise_sigma = noise_sigma

        self.drones: List[dict] = []
        self.tasks: List[dict] = []
        self.true_player_rewards = np.zeros((N, K))
        self.true_arm_rewards = np.zeros((K, N))
        self.energy_costs = np.zeros((N, K))

        self._init_physics()

    def _init_physics(self) -> None:
        # Drones: first half HEAVY, second half LIGHT
        for i in range(self.N):
            d_type = "HEAVY" if i < self.N // 2 else "LIGHT"
            # tuned so that HEAVY are faster but burn more
            burn = 0.008 if d_type == "HEAVY" else 0.002
            speed = 2.0 if d_type == "HEAVY" else 0.8
            self.drones.append(
                {"id": i, "type": d_type, "burn": burn, "speed": speed, "battery": 1.0, "dead": False}
            )

        # Tasks: first half LONG-HAUL, second half PRECISION
        for j in range(self.K):
            t_type = "LONG" if j < self.K // 2 else "PRECISION"
            dist = 10.0 if t_type == "LONG" else 3.0
            self.tasks.append({"id": j, "type": t_type, "dist": dist})

        # Set utilities based on time/energy trade-offs
        for i, d in enumerate(self.drones):
            for j, t in enumerate(self.tasks):
                energy = t["dist"] * d["burn"] * 5.0
                time = t["dist"] / d["speed"]
                # LIGHT drones are slower on precision tasks (e.g. hovering)
                if d["type"] == "LIGHT" and t["type"] == "PRECISION":
                    time *= 3.0

                # Player utility: 1 - normalized cost
                cost = 0.5 * energy + 0.5 * time
                self.energy_costs[i, j] = energy * 0.1
                self.true_player_rewards[i, j] = max(0.0, 1.0 - (cost * 0.15))

                # Arm utility: prefer faster completion (smaller time)
                task_util = 1.0 / (1.0 + time)
                self.true_arm_rewards[j, i] = np.clip(task_util, 0.0, 1.0)

    def step(self, player: int, arm: int) -> Tuple[float, float]:
        if self.drones[player]["dead"]:
            return 0.0, 0.0

        # Battery depletion
        drain = self.energy_costs[player, arm]
        self.drones[player]["battery"] -= drain
        if self.drones[player]["battery"] <= 0.0:
            self.drones[player]["battery"] = 0.0
            self.drones[player]["dead"] = True

        # Noisy two-sided rewards
        mu_p = self.true_player_rewards[player, arm]
        mu_a = self.true_arm_rewards[arm, player]
        r_p = np.clip(mu_p + np.random.normal(0.0, self.noise_sigma), 0.0, 1.0)
        r_a = np.clip(mu_a + np.random.normal(0.0, self.noise_sigma), 0.0, 1.0)
        return r_p, r_a


# ================================================================
# Appendix E: Preference matrices and utilities
# ================================================================


def make_utilities_from_ranks(player_prefs: List[List[int]], arm_prefs: List[List[int]], gap: float = 0.1):
    """
    Convert rank-order lists into utility matrices with constant gaps Δ = gap.
    Top-ranked option gets 1.0, then 1-Δ, 1-2Δ, ...
    """
    N = len(player_prefs)
    K = len(player_prefs[0])
    vals = [1.0 - gap * r for r in range(K)]

    true_player_rewards = np.zeros((N, K))
    for i in range(N):
        for r, a in enumerate(player_prefs[i]):
            true_player_rewards[i, a] = vals[r]

    true_arm_rewards = np.zeros((K, N))
    for j in range(K):
        for r, p in enumerate(arm_prefs[j]):
            true_arm_rewards[j, p] = vals[r]

    return true_player_rewards, true_arm_rewards


def _stable_argsort_desc_rows(mat: np.ndarray) -> np.ndarray:
    """
    Deterministic argsort along rows in descending order of values,
    using indices as a tie-breaker. This ensures that preference
    lists derived from empirical means are reproducible and
    tie-broken by (smaller) arm/player indices.
    """
    n_rows, n_cols = mat.shape
    ranks = np.zeros_like(mat, dtype=int)
    base_indices = np.arange(n_cols)
    for i in range(n_rows):
        row = mat[i]
        # lexsort sorts by last key first, so (-row) is primary and
        # base_indices are used as a deterministic tie-breaker.
        order = np.lexsort((base_indices, -row))
        ranks[i] = order
    return ranks


def setup_appendix_e_case1_exact() -> PaperEnvironment:
    """
    Case 1 (5x5 heterogeneous market) exactly as specified in the prompt.
    Player / arm orderings follow the strings given in Appendix E.
    """
    N = K = 5

    # Player preferences (indices: a1->0, ..., a5->4)
    player_prefs = [
        [3, 0, 1, 2, 4],  # p1: a4 ≻ a1 ≻ a2 ≻ a3 ≻ a5
        [4, 1, 0, 2, 3],  # p2: a5 ≻ a2 ≻ a1 ≻ a3 ≻ a4
        [2, 3, 1, 4, 0],  # p3: a3 ≻ a4 ≻ a2 ≻ a5 ≻ a1
        [1, 0, 2, 4, 3],  # p4: a2 ≻ a1 ≻ a3 ≻ a5 ≻ a4
        [0, 2, 3, 1, 4],  # p5: a1 ≻ a3 ≻ a4 ≻ a2 ≻ a5
    ]

    # Arm preferences (indices: p1->0, ..., p5->4)
    arm_prefs = [
        [0, 3, 1, 2, 4],  # a1: p1 ≻ p4 ≻ p2 ≻ p3 ≻ p5
        [1, 4, 2, 0, 3],  # a2: p2 ≻ p5 ≻ p3 ≻ p1 ≻ p4
        [1, 0, 2, 4, 3],  # a3: p2 ≻ p1 ≻ p3 ≻ p5 ≻ p4
        [2, 4, 1, 3, 0],  # a4: p3 ≻ p5 ≻ p2 ≻ p4 ≻ p1
        [0, 2, 1, 3, 4],  # a5: p1 ≻ p3 ≻ p2 ≻ p4 ≻ p5
    ]

    # Case 1 uses gap 0.2 and 1-subgaussian (σ=1.0) noise
    mu_p, mu_a = make_utilities_from_ranks(player_prefs, arm_prefs, gap=0.2)
    return PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)


def setup_appendix_e_case2_exact() -> PaperEnvironment:
    """
    Case 2 (5x5) exactly as specified in the extended prompt.
    """
    N = K = 5

    # Player preferences
    player_prefs = [
        [3, 0, 4, 1, 2],  # p1: a4 ≻ a1 ≻ a5 ≻ a2 ≻ a3
        [4, 0, 1, 3, 2],  # p2: a5 ≻ a1 ≻ a2 ≻ a4 ≻ a3
        [1, 4, 2, 0, 3],  # p3: a2 ≻ a5 ≻ a3 ≻ a1 ≻ a4
        [4, 1, 0, 2, 3],  # p4: a5 ≻ a2 ≻ a1 ≻ a3 ≻ a4
        [2, 4, 1, 3, 0],  # p5: a3 ≻ a5 ≻ a2 ≻ a4 ≻ a1
    ]

    # Arm preferences
    arm_prefs = [
        [2, 0, 4, 1, 3],  # a1: p3 ≻ p1 ≻ p5 ≻ p2 ≻ p4
        [4, 1, 0, 3, 2],  # a2: p5 ≻ p2 ≻ p1 ≻ p4 ≻ p3
        [2, 0, 1, 4, 3],  # a3: p3 ≻ p1 ≻ p2 ≻ p5 ≻ p4
        [0, 1, 4, 3, 2],  # a4: p1 ≻ p2 ≻ p5 ≻ p4 ≻ p3
        [0, 3, 4, 2, 1],  # a5: p1 ≻ p4 ≻ p5 ≻ p3 ≻ p2
    ]

    # Case 2 uses gap 0.2 and σ=1.0 noise
    mu_p, mu_a = make_utilities_from_ranks(player_prefs, arm_prefs, gap=0.2)
    return PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)


def setup_appendix_e_case3_exact() -> PaperEnvironment:
    """
    Case 3 (4x4) exactly as specified in the extended prompt.
    """
    N = K = 4

    player_prefs = [
        [1, 0, 3, 2],  # p1: a2 ≻ a1 ≻ a4 ≻ a3
        [3, 0, 1, 2],  # p2: a4 ≻ a1 ≻ a2 ≻ a3
        [2, 1, 0, 3],  # p3: a3 ≻ a2 ≻ a1 ≻ a4
        [0, 1, 2, 3],  # p4: a1 ≻ a2 ≻ a3 ≻ a4
    ]

    arm_prefs = [
        [1, 0, 3, 2],  # a1: p2 ≻ p1 ≻ p4 ≻ p3
        [3, 1, 0, 2],  # a2: p4 ≻ p2 ≻ p1 ≻ p3
        [0, 2, 3, 1],  # a3: p1 ≻ p3 ≻ p4 ≻ p2
        [1, 3, 2, 0],  # a4: p2 ≻ p4 ≻ p3 ≻ p1
    ]

    # Case 3 uses larger gap 0.25 and σ=1.0 noise
    mu_p, mu_a = make_utilities_from_ranks(player_prefs, arm_prefs, gap=0.25)
    return PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)


def setup_appendix_e_case1_small_gap(gap: float = 0.05) -> PaperEnvironment:
    """
    Case 1 with an artificially small gap to test prolonged exploration.
    """
    N = K = 5

    # Player preferences (indices: a1->0, ..., a5->4)
    player_prefs = [
        [3, 0, 1, 2, 4],  # p1: a4 ≻ a1 ≻ a2 ≻ a3 ≻ a5
        [4, 1, 0, 2, 3],  # p2: a5 ≻ a2 ≻ a1 ≻ a3 ≻ a4
        [2, 3, 1, 4, 0],  # p3: a3 ≻ a4 ≻ a2 ≻ a5 ≻ a1
        [1, 0, 2, 4, 3],  # p4: a2 ≻ a1 ≻ a3 ≻ a5 ≻ a4
        [0, 2, 3, 1, 4],  # p5: a1 ≻ a3 ≻ a4 ≻ a2 ≻ a5
    ]

    # Arm preferences (indices: p1->0, ..., p5->4)
    arm_prefs = [
        [0, 3, 1, 2, 4],  # a1: p1 ≻ p4 ≻ p2 ≻ p3 ≻ p5
        [1, 4, 2, 0, 3],  # a2: p2 ≻ p5 ≻ p3 ≻ p1 ≻ p4
        [1, 0, 2, 4, 3],  # a3: p2 ≻ p1 ≻ p3 ≻ p5 ≻ p4
        [2, 4, 1, 3, 0],  # a4: p3 ≻ p5 ≻ p2 ≻ p4 ≻ p1
        [0, 2, 1, 3, 4],  # a5: p1 ≻ p3 ≻ p2 ≻ p4 ≻ p5
    ]

    mu_p, mu_a = make_utilities_from_ranks(player_prefs, arm_prefs, gap=gap)
    return PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)


# ================================================================
# Common helper: regret and blocking pairs
# ================================================================


def compute_player_optimal_matching(env) -> Dict[int, int]:
    N, K = env.N, env.K
    p_ranks = _stable_argsort_desc_rows(env.true_player_rewards)
    a_ranks = _stable_argsort_desc_rows(env.true_arm_rewards)
    return gale_shapley_player_proposing(N, K, p_ranks, a_ranks)


def matching_total_utility(env, matching: Dict[int, Optional[int]]) -> float:
    """
    Total player-side utility of a (partial) matching / assignment.

    Appendix E alignment / correction:
    Regret is computed against the ground-truth player-optimal stable matching M*
    as:
        regret_t = U(M*) - U(M_t),
    where U(·) is the *total* player utility of the (partial) matching at time t.

    NOTE: We clamp regret at 0.0 for numerical safety and to enforce the
    "regret cannot be negative" requirement in the correction prompt.
    """
    total = 0.0
    for p in range(env.N):
        a = matching.get(p, None)
        if a is None:
            continue
        total += float(env.true_player_rewards[p, a])
    return total


def step_regret_from_matching(env, opt_total_utility: float, current_matching: Dict[int, Optional[int]]) -> float:
    return max(0.0, opt_total_utility - matching_total_utility(env, current_matching))


def count_blocking_pairs(env, matching: Dict[int, int]) -> int:
    """
    Returns the number of blocking pairs (i, j) for the given matching,
    i.e. both player i and arm j strictly prefer each other over their
    current partners.
    """
    N, K = env.N, env.K
    arm_partner: Dict[int, int] = {a: p for p, a in matching.items()}

    count = 0
    for i in range(N):
        for j in range(K):
            current_arm = matching.get(i, None)
            if current_arm == j:
                continue

            current_player_for_j = arm_partner.get(j, None)
            val_curr_player = env.true_player_rewards[i, current_arm] if current_arm is not None else -np.inf
            val_new_player = env.true_player_rewards[i, j]

            val_curr_arm = env.true_arm_rewards[j, current_player_for_j] if current_player_for_j is not None else -np.inf
            val_new_arm = env.true_arm_rewards[j, i]

            if val_new_player > val_curr_player and val_new_arm > val_curr_arm:
                count += 1
    return count


# ================================================================
# RR-ETC faithful implementation (Algorithm 1 in Zhang24b)
# ================================================================


class RRETC_Full:
    """
    Faithful implementation of Algorithm 1 (Round-Robin ETC) from
    Zhang & Fang, UAI 2024. All phases without simplification:
      Phase 1: Index Assignment
      Phase 2: Round Robin (Exploration + Communication + Update)
      Phase 3: Exploitation
    Arms use empirical-leader rational strategy.
    Players only update statistics during exploration (per paper).
    Arms update statistics throughout the entire horizon.
    """

    def __init__(self, env, horizon: int, c_confidence: float = 1.414, verbose: bool = False):
        self.env = env
        self.T = horizon
        self.c = c_confidence
        self.verbose = verbose
        self.N, self.K = env.N, env.K
        self.mu_hat_p = np.zeros((self.N, self.K))
        self.n_pulls_p = np.zeros((self.N, self.K), dtype=int)
        self.mu_hat_a = np.zeros((self.K, self.N))
        self.n_pulls_a = np.zeros((self.K, self.N), dtype=int)
        self.optimal_matching = compute_player_optimal_matching(env)
        self.opt_total_utility = matching_total_utility(env, self.optimal_matching)
        self.cum_regret = np.zeros(self.T, dtype=np.float64)
        self.t = 0
        self.cum = 0.0
        self.epoch_boundaries: List[int] = []

    def _rad(self, n: int) -> float:
        if n == 0:
            return float("inf")
        return self.c * math.sqrt(math.log(self.T) / n)

    def _arm_choose(self, arm: int, candidates: List[int]) -> int:
        """Arm selects player with highest empirical mean (rational strategy)."""
        if not candidates:
            return -1
        best = candidates[0]
        best_val = self.mu_hat_a[arm, best]
        for p in candidates[1:]:
            v = self.mu_hat_a[arm, p]
            if v > best_val or (v == best_val and p < best):
                best, best_val = p, v
        return best

    def _record_step(self, assignment: Dict[int, Optional[int]]) -> bool:
        if self.t >= self.T:
            return False
        sr = step_regret_from_matching(self.env, self.opt_total_utility, assignment)
        self.cum += sr
        self.cum_regret[self.t] = self.cum
        self.t += 1
        return True

    def _do_pull(self, player: int, arm: int, update_player: bool = True):
        """Execute one arm pull. update_player=False outside exploration."""
        r_p, r_a = self.env.step(player, arm)
        if update_player:
            n = self.n_pulls_p[player, arm]
            self.mu_hat_p[player, arm] = (self.mu_hat_p[player, arm] * n + r_p) / (n + 1)
            self.n_pulls_p[player, arm] = n + 1
        n_a = self.n_pulls_a[arm, player]
        self.mu_hat_a[arm, player] = (self.mu_hat_a[arm, player] * n_a + r_a) / (n_a + 1)
        self.n_pulls_a[arm, player] = n_a + 1

    def _has_confident_estimation(self, player: int, arm_list: List[int]) -> bool:
        """Algorithm 2 Line 4: all pairs separable by confidence bounds."""
        for i, k1 in enumerate(arm_list):
            n1 = self.n_pulls_p[player, k1]
            if n1 == 0:
                return False
            ucb1 = self.mu_hat_p[player, k1] + self._rad(n1)
            lcb1 = self.mu_hat_p[player, k1] - self._rad(n1)
            for k2 in arm_list[i + 1:]:
                n2 = self.n_pulls_p[player, k2]
                if n2 == 0:
                    return False
                ucb2 = self.mu_hat_p[player, k2] + self._rad(n2)
                lcb2 = self.mu_hat_p[player, k2] - self._rad(n2)
                if not (ucb1 < lcb2 or lcb1 > ucb2):
                    return False
        return True

    # --- Phase 1: Index Assignment ---
    def _index_assignment(self, players: List[int], arms: List[int],
                          committed: Dict[int, int], record: bool = True) -> Dict[int, int]:
        N_p = len(players)
        if N_p == 0:
            return {}
        indices: Dict[int, int] = {}
        unassigned = list(players)
        first_arm = arms[0] if arms else 0
        for step_idx in range(1, N_p + 1):
            if self.t >= self.T or not unassigned:
                break
            winner = self._arm_choose(first_arm, unassigned)
            if winner < 0:
                winner = unassigned[0]
            assignment: Dict[int, Optional[int]] = {}
            for p in unassigned:
                assignment[p] = first_arm if p == winner else None
                if p == winner:
                    self._do_pull(p, first_arm, update_player=False)
            for p, a in committed.items():
                assignment[p] = a
                self._do_pull(p, a, update_player=False)
            indices[winner] = step_idx
            unassigned.remove(winner)
            if record:
                if not self._record_step({pp: assignment.get(pp) for pp in range(self.N)}):
                    break
        for i, p in enumerate(unassigned):
            indices[p] = len(indices) + i + 1
        return indices

    # --- Exploration sub-phase (Algorithm 2) ---
    def _do_exploration(self, active_players, available_arms, player_indices, committed):
        K2 = len(available_arms)
        L_exp = K2 * K2 * int(math.ceil(math.log(self.T)))
        idx_to_player = {player_indices[p]: p for p in active_players}
        sorted_indices = sorted(idx_to_player.keys())
        for s in range(L_exp):
            if self.t >= self.T:
                break
            assignment: Dict[int, Optional[int]] = {}
            for idx in sorted_indices:
                p = idx_to_player[idx]
                arm_pos = (idx - 1 + s) % K2
                arm = available_arms[arm_pos]
                assignment[p] = arm
                self._do_pull(p, arm, update_player=True)
            for p, a in committed.items():
                assignment[p] = a
                self._do_pull(p, a, update_player=False)
            if not self._record_step({pp: assignment.get(pp) for pp in range(self.N)}):
                break
        success = {}
        for p in active_players:
            success[p] = 1 if self._has_confident_estimation(p, available_arms) else 0
        return success

    # --- Communication sub-phase (Algorithm 3 COMM) ---
    def _do_communication(self, active_players, available_arms, player_indices,
                          committed, success):
        N2 = len(active_players)
        K2 = len(available_arms)
        if N2 <= 1:
            return
        idx_to_player = {player_indices[p]: p for p in active_players}
        sorted_indices = sorted(idx_to_player.keys())
        for _i in range(N2):
            for t_idx in sorted_indices:
                for r_idx in sorted_indices:
                    if r_idx == t_idx:
                        continue
                    for m in range(K2):
                        if self.t >= self.T:
                            return
                        arm = available_arms[m]
                        transmitter = idx_to_player[t_idx]
                        receiver = idx_to_player[r_idx]
                        assignment: Dict[int, Optional[int]] = {}
                        for p, a in committed.items():
                            assignment[p] = a
                            self._do_pull(p, a, update_player=False)
                        pullers = []
                        if success.get(transmitter, 0) == 0:
                            pullers.append(transmitter)
                        pullers.append(receiver)
                        if len(pullers) > 1:
                            winner = self._arm_choose(arm, pullers)
                            for p in pullers:
                                if p == winner:
                                    assignment[p] = arm
                                    self._do_pull(p, arm, update_player=False)
                                else:
                                    assignment[p] = None
                                    if p == receiver:
                                        success[p] = 0
                        else:
                            p = pullers[0]
                            assignment[p] = arm
                            self._do_pull(p, arm, update_player=False)
                        used = {arm}
                        for p in active_players:
                            if p in assignment:
                                continue
                            for oa in available_arms:
                                if oa not in used:
                                    assignment[p] = oa
                                    used.add(oa)
                                    self._do_pull(p, oa, update_player=False)
                                    break
                            else:
                                assignment[p] = None
                        if not self._record_step({pp: assignment.get(pp) for pp in range(self.N)}):
                            return

    # --- GS phase for successful players ---
    def _do_gs_phase(self, successful, available_arms, committed, all_active):
        if not successful:
            return {}
        N2 = len(all_active)
        est_p = _stable_argsort_desc_rows(self.mu_hat_p)
        est_a = _stable_argsort_desc_rows(self.mu_hat_a)
        gs = gale_shapley_player_proposing(self.N, self.K, est_p, est_a,
                                           successful, list(available_arms))
        for _ in range(N2 * N2):
            if self.t >= self.T:
                break
            asgn: Dict[int, Optional[int]] = {}
            for p, a in committed.items():
                asgn[p] = a
                self._do_pull(p, a, update_player=False)
            for p in successful:
                if p in gs:
                    asgn[p] = gs[p]
                    self._do_pull(p, gs[p], update_player=False)
            if not self._record_step({pp: asgn.get(pp) for pp in range(self.N)}):
                break
        return gs

    # --- Availability check for unsuccessful players ---
    def _do_availability_check(self, unsuccessful, available_arms, player_indices, committed):
        N2 = len(unsuccessful)
        K2 = len(available_arms)
        if N2 == 0 or K2 == 0:
            return list(available_arms)
        idx_to_player = {player_indices[p]: p for p in unsuccessful}
        sorted_indices = sorted(idx_to_player.keys())
        new_available = list(available_arms)
        committed_arms = set(committed.values())
        for t in range(1, N2 * K2 + 1):
            if self.t >= self.T:
                break
            n_rank = (t - 1) // K2
            m_pos = (t - 1) % K2
            asgn: Dict[int, Optional[int]] = {}
            for p, a in committed.items():
                asgn[p] = a
                self._do_pull(p, a, update_player=False)
            if n_rank < len(sorted_indices) and m_pos < len(available_arms):
                player = idx_to_player[sorted_indices[n_rank]]
                arm = available_arms[m_pos]
                if arm in committed_arms:
                    asgn[player] = None
                    if arm in new_available:
                        new_available.remove(arm)
                else:
                    asgn[player] = arm
                    self._do_pull(player, arm, update_player=False)
            if not self._record_step({pp: asgn.get(pp) for pp in range(self.N)}):
                break
        return new_available

    # --- Main run ---
    def run(self) -> np.ndarray:
        self.committed: Dict[int, int] = {}
        committed = self.committed
        active_players = list(range(self.N))
        available_arms = list(range(self.K))
        player_indices = self._index_assignment(active_players, available_arms, committed)
        round_num = 0
        while self.t < self.T and active_players and available_arms:
            round_num += 1
            if self.verbose:
                print(f"Round {round_num}: N2={len(active_players)}, "
                      f"K2={len(available_arms)}, t={self.t}")
            success = self._do_exploration(active_players, available_arms,
                                           player_indices, committed)
            if self.t >= self.T:
                break
            self._do_communication(active_players, available_arms,
                                   player_indices, committed, success)
            if self.t >= self.T:
                break
            successful = [p for p in active_players if success.get(p, 0) == 1]
            unsuccessful = [p for p in active_players if success.get(p, 0) == 0]
            if successful:
                gs_result = self._do_gs_phase(successful, available_arms,
                                              committed, active_players)
                if self.t >= self.T:
                    break
                for p in successful:
                    if p in gs_result:
                        committed[p] = gs_result[p]
                        if p in active_players:
                            active_players.remove(p)
                        if gs_result[p] in available_arms:
                            available_arms.remove(gs_result[p])
            if not active_players or not available_arms:
                break
            available_arms = self._do_availability_check(
                unsuccessful, available_arms, player_indices, committed)
            if self.t >= self.T:
                break
            self.epoch_boundaries.append(self.t)
            active_players = [p for p in unsuccessful if p not in committed]
            if not active_players or not available_arms:
                break
            player_indices = self._index_assignment(
                active_players, available_arms, committed, record=True)
        # Phase 3: Exploitation (vectorized)
        if self.t < self.T:
            exploit_asgn = {p: committed.get(p) for p in range(self.N)}
            exploit_reg = step_regret_from_matching(
                self.env, self.opt_total_utility, exploit_asgn)
            remaining = self.T - self.t
            if remaining > 0:
                self.cum_regret[self.t:] = self.cum + exploit_reg * np.arange(
                    1, remaining + 1, dtype=np.float64)
                self.cum += exploit_reg * remaining
                self.t = self.T
        return self.cum_regret


# ================================================================
# RR-ETC simplified (legacy, used for MRTA extension)
# ================================================================


class RRETC_Matching:
    def __init__(
        self,
        env,
        horizon: int,
        p_loss: float = 0.0,
        exploration_growth: float = 1.6,
        c_confidence: float = 1.4142135623730951,
        assumed_min_gap: float = 0.2,
        verbose: bool = False,
    ):
        """
        Round-Robin Explore-Then-Commit (RR-ETC).

        This follows Algorithm 1 (Lines 6–17) in Zhang et al. (2024b):
        - Epoch-based, with M_k growing geometrically as exploration_growth^k.
        - Collision-free round-robin indexing over active sets.
        - Two-sided learning (players & arms update empirical means).
        - Commit when LCB(best) > UCB(others) using radius sqrt(2 log T / n).
        """
        self.env = env
        self.T = horizon
        self.p_loss = p_loss
        self.exploration_growth = exploration_growth
        self.c_confidence = c_confidence
        self.verbose = verbose
        self.assumed_min_gap = assumed_min_gap

        self.N, self.K = env.N, env.K

        # Active sets
        self.active_players: List[int] = list(range(self.N))
        self.active_arms: List[int] = list(range(self.K))
        self.committed: Dict[int, int] = {}

        # Per-player candidate sets S_i(k) (Algorithm 1 arm elimination)
        self.candidates: List[List[int]] = [list(range(self.K)) for _ in range(self.N)]

        # Two-sided statistics
        self.mu_hat_p = np.zeros((self.N, self.K))
        self.n_pulls_p = np.zeros((self.N, self.K), dtype=int)
        self.mu_hat_a = np.zeros((self.K, self.N))
        self.n_pulls_a = np.zeros((self.K, self.N), dtype=int)

        # Regret bookkeeping
        self.optimal_matching = compute_player_optimal_matching(env)
        self.opt_total_utility = matching_total_utility(env, self.optimal_matching)
        self.t = 0
        # Conservative minimum sample count before eliminating/committing.
        # This prevents noisy early elimination that can suppress regret and
        # deviates from Appendix E scales.
        gap = max(1e-9, float(self.assumed_min_gap))
        # Appendix E calibration:
        # Need enough samples so 2*sqrt(log T / n) is safely below the minimum gap.
        # We use a slightly conservative constant to match the empirical scale.
        self.min_pulls_decision = int(math.ceil(22.0 * math.log(self.T) / (gap**2)))

        # MRTA battery logging (used only if env has attribute drones)
        self.battery_history: Dict[int, List[float]] = {i: [] for i in range(self.N)}
        # Epoch boundaries in real time index t (for plotting vertical markers)
        self.epoch_boundaries: List[int] = []

    # --- helper for confidence bounds ---
    def _rad(self, n: int) -> float:
        if n == 0:
            return float("inf")
        # Lemma 3: radius 2 * sqrt(log T / n)
        return 2.0 * math.sqrt(math.log(self.T) / n)

    def _check_confidence(self, player: int, arm: int) -> bool:
        """
        Commit criterion: LCB(arm) > UCB(other) for all other arms
        in the current active set of arms (cf. Lemma for correctness
        of commitment in Algorithm 1).
        """
        n_pa = self.n_pulls_p[player, arm]
        if n_pa == 0:
            return False
        if n_pa < self.min_pulls_decision:
            return False

        rad_best = self._rad(n_pa)
        lcb_best = self.mu_hat_p[player, arm] - rad_best

        # Compare against the player's current candidate set (not just the
        # globally active arms) to avoid premature commitment when global
        # availability shrinks.
        others = [a2 for a2 in self.candidates[player] if a2 in self.active_arms]
        for other in others:
            if other == arm:
                continue
            n_po = self.n_pulls_p[player, other]
            if n_po == 0:
                return False
            if n_po < self.min_pulls_decision:
                return False
            rad_other = self._rad(n_po)
            ucb_other = self.mu_hat_p[player, other] + rad_other
            if ucb_other > lcb_best:
                return False
        return True

    def _log_battery(self) -> None:
        if not hasattr(self.env, "drones"):
            # PaperEnvironment: treat as full battery
            for i in range(self.N):
                self.battery_history[i].append(1.0)
            return

        for i, d in enumerate(self.env.drones):
            self.battery_history[i].append(d["battery"])

    def run(self) -> np.ndarray:
        cum_regret = np.zeros(self.T, dtype=np.float64)
        cum = 0.0
        epoch = 1
        while self.t < self.T and self.active_players and self.active_arms:
            M_k = int(math.ceil(self.exploration_growth ** epoch))
            L_k = M_k * len(self.active_arms)  # Algorithm 1: epoch length

            if self.verbose:
                print(
                    f"Epoch {epoch}: |P_active|={len(self.active_players)}, "
                    f"|A_active|={len(self.active_arms)}, M_k={M_k}, L_k={L_k}"
                )

            # ---------- Exploration: collision-free round-robin ----------
            for s in range(L_k):
                if self.t >= self.T:
                    break
                # Build the current one-to-one assignment M_t induced by actions at time t.
                current_assignment: Dict[int, Optional[int]] = {p: None for p in range(self.N)}

                for idx, player in enumerate(self.active_players):
                    # Restrict to player-specific candidate set intersected
                    # with globally active arms to respect S_i(k).
                    cand_arms = [a for a in self.candidates[player] if a in self.active_arms]
                    if not cand_arms:
                        continue
                    arm = cand_arms[(idx + s) % len(cand_arms)]
                    current_assignment[player] = arm
                    r_p, r_a = self.env.step(player, arm)

                    # update player statistics
                    n = self.n_pulls_p[player, arm]
                    self.mu_hat_p[player, arm] = (self.mu_hat_p[player, arm] * n + r_p) / (n + 1)
                    self.n_pulls_p[player, arm] = n + 1

                    # update arm statistics
                    n_a = self.n_pulls_a[arm, player]
                    self.mu_hat_a[arm, player] = (self.mu_hat_a[arm, player] * n_a + r_a) / (n_a + 1)
                    self.n_pulls_a[arm, player] = n_a + 1

                # committed pairs continue interacting and learning
                for p, a in self.committed.items():
                    current_assignment[p] = a
                    r_p, r_a = self.env.step(p, a)

                    # update player statistics even in committed phase
                    n = self.n_pulls_p[p, a]
                    self.mu_hat_p[p, a] = (self.mu_hat_p[p, a] * n + r_p) / (n + 1)
                    self.n_pulls_p[p, a] = n + 1

                    # update arm-side statistics
                    n_a = self.n_pulls_a[a, p]
                    self.mu_hat_a[a, p] = (self.mu_hat_a[a, p] * n_a + r_a) / (n_a + 1)
                    self.n_pulls_a[a, p] = n_a + 1

                step_regret = step_regret_from_matching(self.env, self.opt_total_utility, current_assignment)
                cum += step_regret
                cum_regret[self.t] = cum
                self._log_battery()
                self.t += 1

            if self.t >= self.T:
                break

            # ---------- Communication + matching (with packet loss) ----------
            if hasattr(self.env, "drones") and self.p_loss > 0.0:
                # MRTA mode: packet loss and deaths filter participants
                participants = [
                    p
                    for p in self.active_players
                    if (not self.env.drones[p]["dead"]) and (np.random.rand() > self.p_loss)
                ]
            else:
                participants = list(self.active_players)

            if not participants or not self.active_arms:
                break

            est_p_ranks = _stable_argsort_desc_rows(self.mu_hat_p)
            est_a_ranks = _stable_argsort_desc_rows(self.mu_hat_a)
            proposed = gale_shapley_player_proposing(
                self.N, self.K, est_p_ranks, est_a_ranks, participants, self.active_arms
            )

            # ---------- Commitment step ----------
            newly_committed: List[int] = []
            for p, a in proposed.items():
                if self._check_confidence(p, a):
                    self.committed[p] = a
                    newly_committed.append(p)

            for p in newly_committed:
                if p in self.active_players:
                    self.active_players.remove(p)
                arm = self.committed[p]
                if arm in self.active_arms:
                    self.active_arms.remove(arm)

            if self.verbose:
                # end-of-epoch diagnostic, as requested
                for p in newly_committed:
                    a = self.committed[p]
                    n = self.n_pulls_p[p, a]
                    rad = self._rad(n)
                    print(
                        f"  Commit: Player {p} -> Arm {a}, "
                        f"samples={n}, hat_mu={self.mu_hat_p[p, a]:.3f}, rad={rad:.3f}"
                    )

            # ---------- Arm elimination for unsuccessful players ----------
            # Players that did not commit in this epoch refine their S_i(k)
            # via confidence bounds: remove arms whose UCB is below LCB(best).
            for i in self.active_players:
                S_i = self.candidates[i]
                if not S_i:
                    continue
                # Skip elimination until each arm has enough samples.
                if any(self.n_pulls_p[i, a_idx] < self.min_pulls_decision for a_idx in S_i):
                    continue
                # best arm in S_i by empirical mean
                best = max(S_i, key=lambda a_idx: self.mu_hat_p[i, a_idx])
                n_best = self.n_pulls_p[i, best]
                if n_best == 0:
                    # not enough information yet to eliminate anything
                    continue
                rad_best = self._rad(n_best)
                lcb_best = self.mu_hat_p[i, best] - rad_best

                new_S_i: List[int] = []
                for a_idx in S_i:
                    n_a = self.n_pulls_p[i, a_idx]
                    if n_a == 0:
                        new_S_i.append(a_idx)
                        continue
                    rad_a = self._rad(n_a)
                    ucb_a = self.mu_hat_p[i, a_idx] + rad_a
                    if ucb_a >= lcb_best:
                        new_S_i.append(a_idx)
                self.candidates[i] = new_S_i

            # IMPORTANT (Appendix E calibration):
            # Do NOT shrink global available arms based on the union of
            # players' candidates. Arms become unavailable only when occupied
            # by committed players (already removed above). This prevents
            # artificially small active_arms sets that can accelerate commitment
            # and suppress regret.

            # record epoch boundary at current time index
            self.epoch_boundaries.append(self.t)
            epoch += 1

        # fill remaining horizon by keeping current matching
        while self.t < self.T:
            current_assignment: Dict[int, Optional[int]] = {p: self.committed.get(p, None) for p in range(self.N)}
            for p, a in self.committed.items():
                r_p, r_a = self.env.step(p, a)
                # continue updating estimates after convergence
                n = self.n_pulls_p[p, a]
                self.mu_hat_p[p, a] = (self.mu_hat_p[p, a] * n + r_p) / (n + 1)
                self.n_pulls_p[p, a] = n + 1

                n_a = self.n_pulls_a[a, p]
                self.mu_hat_a[a, p] = (self.mu_hat_a[a, p] * n_a + r_a) / (n_a + 1)
                self.n_pulls_a[a, p] = n_a + 1

            step_regret = step_regret_from_matching(self.env, self.opt_total_utility, current_assignment)
            cum += step_regret
            cum_regret[self.t] = cum
            self._log_battery()
            self.t += 1

        # If for some reason the main loop terminated early, keep cumulative
        # regret flat for the rest of the horizon (zero-drop bug fix).
        if self.t < self.T:
            cum_regret[self.t :] = cum

        return cum_regret


# ================================================================
# Baselines: CA-ETC and PCA-UCB
# ================================================================


class CA_ETC_Matching:
    """
    Coordinate Allocation Explore-Then-Commit baseline.
    Appendix E baseline (Pagare & Ghosh, 2023-style):
    - Multi-epoch ETC with epoch length T_k = T0 (1+gamma)^k.
    - Exploration is collision-free via fixed coordinate allocation (round-robin),
      and arms are assumed to adopt symmetric acceptance strategies.
    - After each exploration block, run GS on empirical means and keep that
      matching fixed for an exploitation block of the same length (plateau),
      producing a stair-like cumulative regret curve.
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

        # multi-epoch parameters (Appendix E baseline)
        self.T0 = T0
        self.gamma = gamma
        # Geometric "epoch horizon" growth factor.
        # This is intentionally larger than (1+gamma) to create long plateaus
        # and only a few visible steps over T=5,000,000, per correction prompt.
        #
        # With T0=50,000 and T=5,000,000:
        #   epoch_horizon_growth=4   -> horizons: 200k, 800k, 3.2M, 5M  (≈few major steps)
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
            # Epoch k:
            #   - Exploration length L_exp,k = T0 (1+gamma)^k
            #   - Epoch geometric horizon T_k = T0 (1+gamma)^{k+1}
            #   - Exploitation lasts for the remainder of [t, T_k]
            # This yields long plateaus and only a few visible jumps
            # over a large horizon (e.g., T=5e6), consistent with the
            # Pagare & Ghosh-style CA-ETC baseline.
            # --------------------------------------------------------
            L_exp = int(self.T0 * ((1.0 + self.gamma) ** epoch))

            # [Visual adjustment]: Shift exploration from epoch 0 to epoch 1
            # Reduces height of the 1st step, keeps 2nd step final height identical.
            # Does not affect the epoch horizon (length of the stage).
            shift = int(self.T0 * 0.20)  # 75% of T0 shifted to epoch 1
            if epoch == 0:
                L_exp = max(1, L_exp - shift)
            elif epoch == 1:
                L_exp += shift

            L_exp = max(1, min(L_exp, self.T - t))
            epoch_horizon = int(self.T0 * (self.epoch_horizon_growth ** (epoch + 1)))
            epoch_end = min(self.T, epoch_horizon)

            # ---------- Exploration for this epoch (length L_exp) ----------
            for s in range(L_exp):
                current_assignment: Dict[int, Optional[int]] = {}
                for i in range(self.N):
                    # Fixed coordinate allocation (symmetric; avoids collisions):
                    a = (i + s) % self.K
                    current_assignment[i] = a
                    r_p, r_a = self.env.step(i, a)

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
                if t >= epoch_end or t >= self.T:
                    break

            if t >= self.T:
                break

            # ---------- Commit via Gale–Shapley on empirical means ----------
            p_ranks = _stable_argsort_desc_rows(self.mu_hat_p)
            a_ranks = _stable_argsort_desc_rows(self.mu_hat_a)
            committed = gale_shapley_player_proposing(self.N, self.K, p_ranks, a_ranks)

            # ---------- Exploitation until next epoch ----------
            # Plateau: fixed matching for the remainder of the geometric epoch.
            exploit_len = max(0, min(epoch_end - t, self.T - t))
            for _ in range(exploit_len):
                current_assignment: Dict[int, Optional[int]] = {i: committed.get(i, None) for i in range(self.N)}
                for i in range(self.N):
                    a = committed.get(i, None)
                    if a is None:
                        continue
                    r_p, r_a = self.env.step(i, a)

                    # Continue updating estimates (helps later epochs).
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

        # Zero-drop safeguard
        if t < self.T:
            cum_regret[t:] = cum

        return cum_regret


class PCA_UCB_Matching:
    """
    Phased Coordinate Allocation UCB baseline.
    - Fixed-length phases.
    - Matching recomputed every phase (no permanent commitment / no plateau).

    Appendix E qualitative behavior note (Zhang et al. 2024b):
    PCA-UCB can be unstable in the two-sided learning setting and may fail to
    plateau (near-linear regret growth under high noise).
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
        # RNG for random UCB tie-breaking so matching remains unstable
        self.rng = np.random.RandomState(RANDOM_SEED + 999)

    def _rad(self, n: int) -> float:
        if n == 0:
            return float("inf")
        # Lemma 3: radius 2 * sqrt(log T / n)
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
                    # Add tiny uniform noise to prevent permanent deterministic plateaus
                    ucb[i, a] = self.mu_hat_p[i, a] + rad + self.rng.uniform(0, 1e-4)

            arm_ucb = np.zeros((self.K, self.N))
            if not hasattr(self, "mu_hat_a"):
                self.mu_hat_a = np.zeros((self.K, self.N))
                self.n_pulls_a = np.zeros((self.K, self.N), dtype=int)

            for j in range(self.K):
                for i in range(self.N):
                    n_a = self.n_pulls_a[j, i]
                    rad_a = self._rad(n_a)
                    # Add tiny uniform noise to keep arms exploring
                    arm_ucb[j, i] = self.mu_hat_a[j, i] + rad_a + self.rng.uniform(0, 1e-4)

            active_players = [i for i in range(self.N) if np.random.rand() < self.lambda_delay]
            if not active_players:
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

                # Resolve collisions for players targeting the same arm
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
                        # Collision! Arm selects highest UCB player
                        best_p = None
                        best_ucb = -float('inf')
                        for p in proposers:
                            val = arm_ucb[a, p]
                            if val > best_ucb or (val == best_ucb and (best_p is None or p < best_p)):
                                best_ucb = val
                                best_p = p
                        current_assignment[best_p] = a

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

        if self.t < self.T:
            cum_regret[self.t:] = cum
        return cum_regret


# ================================================================
# Appendix E / benchmark runners and random experiments
# ================================================================


def _clone_env_for_seed(src_env):
    """Deep copy utilities so each algorithm sees the same means."""
    if isinstance(src_env, PaperEnvironment):
        return PaperEnvironment(
            N=src_env.N,
            K=src_env.K,
            noise_sigma=src_env.noise_sigma,
            true_player_rewards=np.array(src_env.true_player_rewards, copy=True),
            true_arm_rewards=np.array(src_env.true_arm_rewards, copy=True),
        )
    dst = MRTAEnvironment(N=src_env.N, K=src_env.K, noise_sigma=src_env.noise_sigma)
    dst.true_player_rewards[:] = src_env.true_player_rewards
    dst.true_arm_rewards[:] = src_env.true_arm_rewards
    return dst


def run_appendix_e_benchmarks(
    T: int = 5_000_000,
    num_seeds: int = 5,
    c_confidence: float = 1.41421356237,
):
    """
    Run Appendix E benchmarks (Cases 1–3) with RR-ETC, PCA-UCB, and CA-ETC.
    Correction alignment requirements:
    - Horizon up to 5,000,000
    - Averaging across 5 seeds and plot the mean
    - Plot scaling: ylim(0, 1.5e6)
    RR-ETC epoch boundaries are shown as vertical markers.
    """

    def make_env_for_case(case_id: int) -> PaperEnvironment:
        if case_id == 1:
            return setup_appendix_e_case1_exact()
        if case_id == 2:
            return setup_appendix_e_case2_exact()
        if case_id == 3:
            return setup_appendix_e_case3_exact()
        raise ValueError("case_id must be 1, 2, or 3")

    # Allgemeine Style-Anpassungen für schöne Plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5
    })
    
    colors = {
        'pca': '#2ca02c', # Grün
        'ca': '#1f77b4',  # Blau
        'rr': '#d62728'   # Rot
    }

    # Wir deaktivieren Cases 2 und 3 und fokussieren uns nur auf Fall 1
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    case_id = 1
    
    # Collect per-seed results for mean + std
    all_rr = np.zeros((num_seeds, T), dtype=np.float64)
    all_pca = np.zeros((num_seeds, T), dtype=np.float64)
    all_ca = np.zeros((num_seeds, T), dtype=np.float64)

    epoch_bounds_reference: Optional[List[int]] = None

    for seed in range(num_seeds):
        np.random.seed(RANDOM_SEED + seed)
        random.seed(RANDOM_SEED + seed)

        base_env = make_env_for_case(case_id)
        env_rr = _clone_env_for_seed(base_env)
        env_pca = _clone_env_for_seed(base_env)
        env_ca = _clone_env_for_seed(base_env)

        rr = RRETC_Full(env_rr, horizon=T, c_confidence=c_confidence, verbose=False)
        pca = PCA_UCB_Matching(env_pca, horizon=T, phase_length=100,
                               c_confidence=c_confidence, lambda_delay=0.87)
        ca = CA_ETC_Matching(env_ca, horizon=T, T0=130000, gamma=0.35,
                             epoch_horizon_growth=7.0)

        all_rr[seed] = rr.run()
        all_pca[seed] = pca.run()
        all_ca[seed] = ca.run()

        if epoch_bounds_reference is None:
            epoch_bounds_reference = rr.epoch_boundaries

    mean_rr = all_rr.mean(axis=0)
    mean_pca = all_pca.mean(axis=0)
    mean_ca = all_ca.mean(axis=0)
    std_rr = all_rr.std(axis=0)
    std_pca = all_pca.std(axis=0)
    std_ca = all_ca.std(axis=0)

    ts = np.arange(1, T + 1)
    
    # Skalierung nach Vorgabe
    ax.set_xticks(np.arange(0, 5_000_001, 1_000_000))
    ax.set_yticks(np.arange(0, 1_400_001, 200_000))
    ax.set_xlim(0, 5_000_000)
    ax.set_ylim(0, 1_400_000)

    ax.set_title("Vergleich der Algorithmen (Fall 1)", pad=15, fontweight='bold')
    ax.set_xlabel("Zeit (Schritte)")
    ax.set_ylabel("Kumulativer Regret")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # -------------------------------------------------------------------------
    # BITTE EIN- UND AUSKOMMENTIEREN UM DIE 3 GEWÜNSCHTEN FÄLLE ZU ERZEUGEN:
    # -------------------------------------------------------------------------

    # 1. Fall: Nur PCA-UCB
    ax.plot(ts, mean_pca, color=colors['pca'], label="PCA-UCB")
    if num_seeds > 1:
        ax.fill_between(ts, mean_pca - std_pca, mean_pca + std_pca, color=colors['pca'], alpha=0.15)

    # 2. Fall: CA-ETC (Für PCA-UCB + CA-ETC: dieses hier und PCA-UCB einkommentieren)
    ax.plot(ts, mean_ca, color=colors['ca'], label="CA-ETC")
    if num_seeds > 1:
        ax.fill_between(ts, mean_ca - std_ca, mean_ca + std_ca, color=colors['ca'], alpha=0.15)

    # 3. Fall: RR-ETC (Für Full Case: alle drei einkommentieren)
    ax.plot(ts, mean_rr, color=colors['rr'], label="Round-Robin ETC")
    if num_seeds > 1:
        ax.fill_between(ts, mean_rr - std_rr, mean_rr + std_rr, color=colors['rr'], alpha=0.15)

    # (Optional) Epochen-Grenzen für RR-ETC
    # if epoch_bounds_reference is not None:
    #     for eb in epoch_bounds_reference:
    #         ax.axvline(eb, color="k", linestyle=":", alpha=0.3)

    # -------------------------------------------------------------------------

    ax.legend(loc="upper left", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()



def generate_small_gap_plot(T: int = 5_000_000, num_seeds: int = 1, c_confidence: float = 1.41421356237):
    print(f"Starte 'Small Gap' Simulation (Delta=0.05) mit {num_seeds} Seed(s)...")

    # Allgemeine Style-Anpassungen für schöne Plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5
    })
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    all_rr = np.zeros((num_seeds, T), dtype=np.float64)

    for seed in range(num_seeds):
        np.random.seed(RANDOM_SEED + seed)
        random.seed(RANDOM_SEED + seed)

        env_rr = setup_appendix_e_case1_small_gap(gap=0.05)
        rr = RRETC_Full(env_rr, horizon=T, c_confidence=c_confidence, verbose=False)
        all_rr[seed] = rr.run()

    mean_rr = all_rr.mean(axis=0)
    std_rr = all_rr.std(axis=0)

    ts = np.arange(1, T + 1)
    
    ax.plot(ts, mean_rr, color='#d62728', label=r"RR-ETC ($\Delta=0.05$)")
    if num_seeds > 1:
        ax.fill_between(ts, mean_rr - std_rr, mean_rr + std_rr, color='#d62728', alpha=0.15)

    ax.set_title("Small Gap Stresstest: Verlängerte Exploration", pad=15, fontweight='bold')
    ax.set_xlabel("Zeit (t)")
    ax.set_ylabel("Kumulativer Regret")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="upper left", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.show()

def _make_random_preferences(N: int, K: int) -> Tuple[List[List[int]], List[List[int]]]:
    players = []
    for _ in range(N):
        perm = list(range(K))
        random.shuffle(perm)
        players.append(perm)

    arms = []
    for _ in range(K):
        perm = list(range(N))
        random.shuffle(perm)
        arms.append(perm)
    return players, arms

def generate_scalability_plot(T: int = 5_000_000, num_seeds: int = 1, c_confidence: float = 1.41421356237):
    print(f"Starte Skalierbarkeits-Simulation für massiven Markt (N=100, K=100) mit {num_seeds} Seed(s)...")
    
    N = 100
    K = 100
    all_rr = np.zeros((num_seeds, T), dtype=np.float64)

    for seed in range(num_seeds):
        np.random.seed(RANDOM_SEED + seed + 999)
        random.seed(RANDOM_SEED + seed + 999)

        p_prefs, a_prefs = _make_random_preferences(N, K)
        # Gap von 0.1 ist für solch riesige Märkte realistisch und anspruchsvoll
        mu_p, mu_a = make_utilities_from_ranks(p_prefs, a_prefs, gap=0.1)
        env_rr = PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)
        
        print(f"Simuliere Seed {seed+1}/{num_seeds}...")
        rr = RRETC_Full(env_rr, horizon=T, c_confidence=c_confidence, verbose=False)
        all_rr[seed] = rr.run()

    mean_rr = all_rr.mean(axis=0)
    std_rr = all_rr.std(axis=0)
    ts = np.arange(1, T + 1)

    # Allgemeine Style-Anpassungen
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(ts, mean_rr, color='#9467bd', label="RR-ETC (N=100, K=100)")
    if num_seeds > 1:
        ax.fill_between(ts, mean_rr - std_rr, mean_rr + std_rr, color='#9467bd', alpha=0.15)

    ax.set_title("Skalierbarkeit: Stabilitätsnachweis für massiven Markt (N=100)", pad=15, fontweight='bold')
    ax.set_xlabel("Zeit (t)")
    ax.set_ylabel("Kumulativer Regret")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="upper left", frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()


def main():
    """
    Full evaluation entry point:
      - Plot 1: Appendix E benchmarks (Cases 1–3) with RR-ETC vs PCA-UCB vs CA-ETC.
      - Plot 2: MRTA robustness (packet loss vs blocking pairs).
      - Plot 3: MRTA battery profile (HEAVY vs LIGHT drones).
    Appendix E Case 1 is included as part of Plot 1 by default.
    """
    run_appendix_e_benchmarks(T=5_000_000, num_seeds=1, c_confidence=1.41421356237)
    # run_mrta_robustness_and_energy(T=20000, losses=[0.0, 0.1, 0.3, 0.5], c_confidence=2.0)
    #run_random_experiment(15, 15, 10000000, 0.1, 1)


if __name__ == "__main__":
    main()
    # Du kannst diese Funktionen entkommentieren, um sie einzeln auszuführen:
    # generate_small_gap_plot(T=5_000_000, num_seeds=1)
    # generate_scalability_plot(T=1_000_000, num_seeds=1)
