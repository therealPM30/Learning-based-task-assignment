import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ================================================================
# Global config / Random Seed Settings
# ================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ================================================================
# Utility: Gale–Shapley stable matching
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
    Standard Gale-Shapley algorithm where players propose to arms.

    Args:
        N: Total number of players.
        K: Total number of arms.
        player_ranks: player_ranks[i] is a permutation of arms (descending utilities).
        arm_ranks: arm_ranks[j] is a permutation of players (descending utilities).
        active_players: List of subset of players currently participating.
        active_arms: List of subset of arms currently available.

    Returns:
        A dictionary mapping player indices to matched arm indices.
    """
    if active_players is None:
        active_players = list(range(N))
    if active_arms is None:
        active_arms = list(range(K))

    free_players = list(active_players)
    next_proposal_index = {i: 0 for i in active_players}
    arm_partner: Dict[int, int] = {}
    player_partner: Dict[int, int] = {}

    # Precompute inverse ranks for arms to determine preference efficiently:
    # arm_pref_order[j][i] = rank of player i for arm j (0 is best)
    arm_pref_order = np.empty((K, N), dtype=int)
    for j in range(K):
        for r, i in enumerate(arm_ranks[j]):
            arm_pref_order[j, i] = r

    while free_players:
        i = free_players.pop(0)
        # Skip if the player is already matched
        if i in player_partner:
            continue

        # Find the next arm in player i's preference list that is currently active
        while next_proposal_index[i] < K:
            a = player_ranks[i, next_proposal_index[i]]
            next_proposal_index[i] += 1
            if a in active_arms:
                break
        else:
            # Exhausted all active arms
            continue

        current_partner = arm_partner.get(a)
        if current_partner is None:
            # Arm is free, temporarily assign
            arm_partner[a] = i
            player_partner[i] = a
        else:
            # Arm decides based on its own preference
            if arm_pref_order[a, i] < arm_pref_order[a, current_partner]:
                # Arm prefers the new proposing player
                arm_partner[a] = i
                player_partner[i] = a
                del player_partner[current_partner]
                free_players.append(current_partner)
            else:
                # Arm rejects the new proposing player
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
    Gale-Shapley algorithm where ARMS propose to players (arm-optimal stable matching).

    Returns:
        A dictionary mapping player indices to arm indices for the matched pairs.
    """
    if active_players is None:
        active_players = list(range(N))
    if active_arms is None:
        active_arms = list(range(K))

    free_arms = list(active_arms)
    next_proposal_index = {a: 0 for a in active_arms}
    player_partner: Dict[int, int] = {}
    arm_partner: Dict[int, int] = {}

    # Precompute inverse ranks for players:
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

        # Find the next player in arm a's preference list that is currently active
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
            # Player decides based on their preference
            if player_pref_order[i, a] < player_pref_order[i, current_arm_for_i]:
                # Player prefers the new proposing arm
                player_partner[i] = a
                arm_partner[a] = i
                del arm_partner[current_arm_for_i]
                free_arms.append(current_arm_for_i)
            else:
                free_arms.append(a)

    return player_partner


# ================================================================
# Environments
#   - PaperEnvironment: Appendix E reproduction (Standard MAB)
#   - MRTAEnvironment: Multi-robot extension (Battery, distances)
# ================================================================

@dataclass
class PaperEnvironment:
    """Standard generic environment based on Zhang et al. (2024)."""
    N: int
    K: int
    noise_sigma: float = 0.1
    true_player_rewards: np.ndarray = None  # shape (N, K)
    true_arm_rewards: np.ndarray = None  # shape (K, N)

    def step(self, player: int, arm: int) -> Tuple[float, float]:
        """Provides two-sided Gaussian noise rewards."""
        mu_p = self.true_player_rewards[player, arm]
        mu_a = self.true_arm_rewards[arm, player]
        r_p = mu_p + np.random.normal(0.0, self.noise_sigma)
        r_a = mu_a + np.random.normal(0.0, self.noise_sigma)
        return r_p, r_a


class MRTAEnvironment:
    """
    MRTA extension environment simulating drones.
    Features heterogeneous drones (HEAVY/LIGHT), heterogeneous tasks,
    and linear battery depletion based on distance and speed.
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
        # Initializing drones: first half HEAVY, second half LIGHT
        for i in range(self.N):
            d_type = "HEAVY" if i < self.N // 2 else "LIGHT"
            # Tuned settings: HEAVY drones are faster but consume more battery
            burn = 0.008 if d_type == "HEAVY" else 0.002
            speed = 2.0 if d_type == "HEAVY" else 0.8
            self.drones.append(
                {"id": i, "type": d_type, "burn": burn, "speed": speed, "battery": 1.0, "dead": False}
            )

        # Initializing tasks: first half LONG-HAUL, second half PRECISION
        for j in range(self.K):
            t_type = "LONG" if j < self.K // 2 else "PRECISION"
            dist = 10.0 if t_type == "LONG" else 3.0
            self.tasks.append({"id": j, "type": t_type, "dist": dist})

        # Calculate utilities based on time and energy trade-offs
        for i, d in enumerate(self.drones):
            for j, t in enumerate(self.tasks):
                energy = t["dist"] * d["burn"] * 5.0
                time = t["dist"] / d["speed"]
                
                # Penalize lightweight drones on precision tasks (e.g., hovering requirements)
                if d["type"] == "LIGHT" and t["type"] == "PRECISION":
                    time *= 3.0

                # Player (Drone) utility: 1 minus a normalized cost function
                cost = 0.5 * energy + 0.5 * time
                self.energy_costs[i, j] = energy * 0.1
                self.true_player_rewards[i, j] = max(0.0, 1.0 - (cost * 0.15))

                # Arm (Task) utility: Preference for faster completion times
                task_util = 1.0 / (1.0 + time)
                self.true_arm_rewards[j, i] = np.clip(task_util, 0.0, 1.0)

    def step(self, player: int, arm: int) -> Tuple[float, float]:
        if self.drones[player]["dead"]:
            return 0.0, 0.0

        # Implement battery depletion mechanics
        drain = self.energy_costs[player, arm]
        self.drones[player]["battery"] -= drain
        if self.drones[player]["battery"] <= 0.0:
            self.drones[player]["battery"] = 0.0
            self.drones[player]["dead"] = True

        # Provide noisy two-sided rewards matching environment characteristics
        mu_p = self.true_player_rewards[player, arm]
        mu_a = self.true_arm_rewards[arm, player]
        r_p = np.clip(mu_p + np.random.normal(0.0, self.noise_sigma), 0.0, 1.0)
        r_a = np.clip(mu_a + np.random.normal(0.0, self.noise_sigma), 0.0, 1.0)
        return r_p, r_a


# ================================================================
# Appendix E: Setup helpers and matrix generation
# ================================================================

def make_utilities_from_ranks(player_prefs: List[List[int]], arm_prefs: List[List[int]], gap: float = 0.1):
    """
    Convert provided rank-order preference lists into precise utility matrices.
    Applies constant gaps (Δ = gap). The top-ranked option yields an expected reward of 1.0, 
    the second 1-Δ, the third 1-2Δ, and so on.
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
    Produces a deterministic argsort along rows in descending order of values.
    Uses positional indices as a tie-breaker. This ensures that preference lists 
    derived from empirical means are reproducible and objectively tie-broken by 
    (smaller) arm or player index values.
    """
    n_rows, n_cols = mat.shape
    ranks = np.zeros_like(mat, dtype=int)
    base_indices = np.arange(n_cols)
    for i in range(n_rows):
        row = mat[i]
        # Using lexsort: sorts by last key first. Negative row (-row) establishes primary 
        # descending sort order, while base_indices provides a deterministic tie-breaker.
        order = np.lexsort((base_indices, -row))
        ranks[i] = order
    return ranks


def setup_appendix_e_case1_exact() -> PaperEnvironment:
    """
    Recreates Case 1 (5x5 heterogeneous market) identical to the experimental specifics.
    Player and arm orderings match those given in Appendix E.
    """
    N = K = 5

    # Player preferences (arm indices map: a1->0, ..., a5->4)
    player_prefs = [
        [3, 0, 1, 2, 4],  # p1: a4 ≻ a1 ≻ a2 ≻ a3 ≻ a5
        [4, 1, 0, 2, 3],  # p2: a5 ≻ a2 ≻ a1 ≻ a3 ≻ a4
        [2, 3, 1, 4, 0],  # p3: a3 ≻ a4 ≻ a2 ≻ a5 ≻ a1
        [1, 0, 2, 4, 3],  # p4: a2 ≻ a1 ≻ a3 ≻ a5 ≻ a4
        [0, 2, 3, 1, 4],  # p5: a1 ≻ a3 ≻ a4 ≻ a2 ≻ a5
    ]

    # Arm preferences (player indices map: p1->0, ..., p5->4)
    arm_prefs = [
        [0, 3, 1, 2, 4],  # a1: p1 ≻ p4 ≻ p2 ≻ p3 ≻ p5
        [1, 4, 2, 0, 3],  # a2: p2 ≻ p5 ≻ p3 ≻ p1 ≻ p4
        [1, 0, 2, 4, 3],  # a3: p2 ≻ p1 ≻ p3 ≻ p5 ≻ p4
        [2, 4, 1, 3, 0],  # a4: p3 ≻ p5 ≻ p2 ≻ p4 ≻ p1
        [0, 2, 1, 3, 4],  # a5: p1 ≻ p3 ≻ p2 ≻ p4 ≻ p5
    ]

    # Case 1 applies a utility gap of 0.2 and 1-subgaussian (σ=1.0) noise
    mu_p, mu_a = make_utilities_from_ranks(player_prefs, arm_prefs, gap=0.2)
    return PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)


def setup_appendix_e_case2_exact() -> PaperEnvironment:
    """
    Recreates Case 2 (5x5) identical to experimental specifics.
    """
    N = K = 5

    player_prefs = [
        [3, 0, 4, 1, 2],  # p1: a4 ≻ a1 ≻ a5 ≻ a2 ≻ a3
        [4, 0, 1, 3, 2],  # p2: a5 ≻ a1 ≻ a2 ≻ a4 ≻ a3
        [1, 4, 2, 0, 3],  # p3: a2 ≻ a5 ≻ a3 ≻ a1 ≻ a4
        [4, 1, 0, 2, 3],  # p4: a5 ≻ a2 ≻ a1 ≻ a3 ≻ a4
        [2, 4, 1, 3, 0],  # p5: a3 ≻ a5 ≻ a2 ≻ a4 ≻ a1
    ]

    arm_prefs = [
        [2, 0, 4, 1, 3],  # a1: p3 ≻ p1 ≻ p5 ≻ p2 ≻ p4
        [4, 1, 0, 3, 2],  # a2: p5 ≻ p2 ≻ p1 ≻ p4 ≻ p3
        [2, 0, 1, 4, 3],  # a3: p3 ≻ p1 ≻ p2 ≻ p5 ≻ p4
        [0, 1, 4, 3, 2],  # a4: p1 ≻ p2 ≻ p5 ≻ p4 ≻ p3
        [0, 3, 4, 2, 1],  # a5: p1 ≻ p4 ≻ p5 ≻ p3 ≻ p2
    ]

    # Case 2 applies a utility gap of 0.2 and (σ=1.0) noise
    mu_p, mu_a = make_utilities_from_ranks(player_prefs, arm_prefs, gap=0.2)
    return PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)


def setup_appendix_e_case3_exact() -> PaperEnvironment:
    """
    Recreates Case 3 (4x4) identical to experimental specifics.
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

    # Case 3 applies a larger utility gap of 0.25 and (σ=1.0) noise
    mu_p, mu_a = make_utilities_from_ranks(player_prefs, arm_prefs, gap=0.25)
    return PaperEnvironment(N=N, K=K, noise_sigma=1.0, true_player_rewards=mu_p, true_arm_rewards=mu_a)


# ================================================================
# Common Metrics & Regret Helpers
# ================================================================

def compute_player_optimal_matching(env) -> Dict[int, int]:
    """Calculates the baseline player-optimal stable matching."""
    N, K = env.N, env.K
    p_ranks = _stable_argsort_desc_rows(env.true_player_rewards)
    a_ranks = _stable_argsort_desc_rows(env.true_arm_rewards)
    return gale_shapley_player_proposing(N, K, p_ranks, a_ranks)


def matching_total_utility(env, matching: Dict[int, Optional[int]]) -> float:
    """
    Computes total player-side utility of a given (partial) matching.
    
    Notes:
    Regret is calculated against the ground-truth player-optimal stable matching M*.
    We measure this as: regret_t = U(M*) - U(M_t)
    where U(·) is the net player utility.
    """
    total = 0.0
    for p in range(env.N):
        a = matching.get(p, None)
        if a is None:
            continue
        total += float(env.true_player_rewards[p, a])
    return total


def step_regret_from_matching(env, opt_total_utility: float, current_matching: Dict[int, Optional[int]]) -> float:
    """Calculates step regret clamped at 0.0 to prevent negative regret values."""
    return max(0.0, opt_total_utility - matching_total_utility(env, current_matching))


def count_blocking_pairs(env, matching: Dict[int, int]) -> int:
    """
    Identifies the amount of blocking pairs (i, j) for the active matching.
    A blocking pair exists if both player i and arm j strictly prefer one another over their current partners.
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
