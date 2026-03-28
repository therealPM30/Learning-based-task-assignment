"""
Faithful implementation of Algorithm 1 (Round-Robin ETC) from
Zhang & Fang, "Decentralized Two-Sided Bandit Learning in Matching Market", UAI 2024.

Includes ALL phases without simplification:
  Phase 1: Index Assignment
  Phase 2: Round Robin (Exploration + Communication + Update)
  Phase 3: Exploitation
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# These are imported from the main module when integrated
from RRETC import (
    _stable_argsort_desc_rows,
    compute_player_optimal_matching,
    gale_shapley_player_proposing,
    matching_total_utility,
    step_regret_from_matching,
)


class RRETC_Full:
    """
    Faithful implementation of Algorithm 1 (Round-Robin ETC).

    All three phases are implemented:
      1. Index Assignment (Algorithm 1, Line 1)
      2. Round Robin with sub-phases (Algorithm 1, Lines 3-18):
         - Exploration (Algorithm 2)
         - Communication (Algorithm 3 COMM)
         - Update (GS matching + availability check + re-indexing)
      3. Exploitation (Algorithm 1, Line 19)

    Arms use the empirical-leader rational strategy (R=16).
    Players only update statistics during exploration (per paper).
    Arms update statistics throughout the entire horizon.
    """

    def __init__(
        self,
        env,
        horizon: int,
        c_confidence: float = 1.3,
        verbose: bool = False,
    ):
        self.env = env
        self.T = horizon
        self.c = c_confidence
        self.verbose = verbose
        self.N, self.K = env.N, env.K

        # Two-sided statistics
        self.mu_hat_p = np.zeros((self.N, self.K))
        self.n_pulls_p = np.zeros((self.N, self.K), dtype=int)
        self.mu_hat_a = np.zeros((self.K, self.N))
        self.n_pulls_a = np.zeros((self.K, self.N), dtype=int)

        # Regret bookkeeping
        self.optimal_matching = compute_player_optimal_matching(env)
        self.opt_total_utility = matching_total_utility(env, self.optimal_matching)
        self.cum_regret = np.zeros(self.T, dtype=np.float64)
        self.t = 0
        self.cum = 0.0
        self.epoch_boundaries: List[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rad(self, n: int) -> float:
        if n == 0:
            return float("inf")
        return self.c * math.sqrt(math.log(self.T) / n)

    def _arm_choose(self, arm: int, candidates: List[int]) -> int:
        """Arm selects player with highest empirical mean (rational strategy).
        Tie-break: smaller player index (deterministic)."""
        if not candidates:
            return -1
        best = candidates[0]
        best_val = self.mu_hat_a[arm, best]
        for p in candidates[1:]:
            v = self.mu_hat_a[arm, p]
            if v > best_val or (v == best_val and p < best):
                best = p
                best_val = v
        return best

    def _record_step(self, assignment: Dict[int, Optional[int]]) -> bool:
        """Record one time step of regret. Returns False if horizon reached."""
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
            self.mu_hat_p[player, arm] = (
                self.mu_hat_p[player, arm] * n + r_p
            ) / (n + 1)
            self.n_pulls_p[player, arm] = n + 1
        # Arms always update (paper: "arms continuously update throughout T")
        n_a = self.n_pulls_a[arm, player]
        self.mu_hat_a[arm, player] = (
            self.mu_hat_a[arm, player] * n_a + r_a
        ) / (n_a + 1)
        self.n_pulls_a[arm, player] = n_a + 1

    def _has_confident_estimation(self, player: int, arm_list: List[int]) -> bool:
        """Check confident estimation (Algorithm 2, Line 4):
        For all k1 != k2 in arm_list, either UCB(k1)<LCB(k2) or LCB(k1)>UCB(k2)."""
        for i, k1 in enumerate(arm_list):
            n1 = self.n_pulls_p[player, k1]
            if n1 == 0:
                return False
            ucb1 = self.mu_hat_p[player, k1] + self._rad(n1)
            lcb1 = self.mu_hat_p[player, k1] - self._rad(n1)
            for k2 in arm_list[i + 1 :]:
                n2 = self.n_pulls_p[player, k2]
                if n2 == 0:
                    return False
                ucb2 = self.mu_hat_p[player, k2] + self._rad(n2)
                lcb2 = self.mu_hat_p[player, k2] - self._rad(n2)
                if not (ucb1 < lcb2 or lcb1 > ucb2):
                    return False
        return True

    # ------------------------------------------------------------------
    # Phase 1: Index Assignment
    # ------------------------------------------------------------------

    def _index_assignment(
        self,
        players: List[int],
        arms: List[int],
        committed: Dict[int, int],
        record: bool = True,
    ) -> Dict[int, int]:
        """INDEX-ASSIGNMENT procedure: assign indices 1..len(players).
        Each step, all unassigned players pull arms[0]. Arm accepts one.
        Accepted player gets index = step number and moves to arms[1].
        Costs len(players) time steps."""
        N_p = len(players)
        if N_p == 0:
            return {}
        indices: Dict[int, int] = {}
        unassigned = list(players)
        first_arm = arms[0] if arms else 0
        second_arm = arms[1] if len(arms) > 1 else arms[0] if arms else 0

        for step_idx in range(1, N_p + 1):
            if self.t >= self.T or not unassigned:
                break
            # All unassigned pull first_arm
            winner = self._arm_choose(first_arm, unassigned)
            if winner < 0:
                winner = unassigned[0]

            assignment: Dict[int, Optional[int]] = {}
            for p in unassigned:
                if p == winner:
                    assignment[p] = first_arm
                    self._do_pull(p, first_arm, update_player=False)
                else:
                    # Rejected players still pull an arm (first_arm in paper)
                    assignment[p] = None
            # Already-indexed players from this batch pull second_arm
            for p in players:
                if p not in unassigned and p not in committed:
                    assignment[p] = second_arm
                    self._do_pull(p, second_arm, update_player=False)
            # Committed players keep their arms
            for p, a in committed.items():
                assignment[p] = a
                self._do_pull(p, a, update_player=False)

            indices[winner] = step_idx
            unassigned.remove(winner)

            if record:
                full_asgn = {p: assignment.get(p) for p in range(self.N)}
                if not self._record_step(full_asgn):
                    break

        # Assign remaining if horizon hit early
        for i, p in enumerate(unassigned):
            indices[p] = step_idx + i + 1 if 'step_idx' in dir() else i + 1

        return indices

    # ------------------------------------------------------------------
    # Phase 2: Round Robin sub-phases
    # ------------------------------------------------------------------

    def _do_exploration(
        self,
        active_players: List[int],
        available_arms: List[int],
        player_indices: Dict[int, int],
        committed: Dict[int, int],
    ) -> Dict[int, int]:
        """Exploration sub-phase (Algorithm 2).
        Length: K2^2 * ceil(log T) time steps.
        Collision-free round-robin: player with index i at step s
        pulls arm (i-1 + s) mod K2 in available_arms."""
        K2 = len(available_arms)
        L_exp = K2 * K2 * int(math.ceil(math.log(self.T)))

        # Sort active players by their index for round-robin ordering
        idx_to_player = {}
        for p in active_players:
            idx_to_player[player_indices[p]] = p
        sorted_indices = sorted(idx_to_player.keys())

        success: Dict[int, int] = {}

        for s in range(L_exp):
            if self.t >= self.T:
                break
            assignment: Dict[int, Optional[int]] = {}

            # Active players explore via collision-free round-robin
            for rank, idx in enumerate(sorted_indices):
                p = idx_to_player[idx]
                arm_pos = (idx - 1 + s) % K2
                arm = available_arms[arm_pos]
                assignment[p] = arm
                self._do_pull(p, arm, update_player=True)  # exploration: update player

            # Committed players keep pulling their arms
            for p, a in committed.items():
                assignment[p] = a
                self._do_pull(p, a, update_player=False)

            full_asgn = {pp: assignment.get(pp) for pp in range(self.N)}
            if not self._record_step(full_asgn):
                break

        # After exploration, check confident estimation for each player
        for p in active_players:
            success[p] = 1 if self._has_confident_estimation(p, available_arms) else 0

        return success

    def _do_communication(
        self,
        active_players: List[int],
        available_arms: List[int],
        player_indices: Dict[int, int],
        committed: Dict[int, int],
        success: Dict[int, int],
    ) -> None:
        """Communication sub-phase (Algorithm 3 COMM).
        Pairwise communication through deliberate conflicts.
        Total steps: N2 * N2 * (N2-1) * K2."""
        N2 = len(active_players)
        K2 = len(available_arms)
        if N2 <= 1:
            return  # No communication needed with 0 or 1 player

        # Build index-to-player mapping
        idx_to_player: Dict[int, int] = {}
        for p in active_players:
            idx_to_player[player_indices[p]] = p
        sorted_indices = sorted(idx_to_player.keys())

        # Algorithm 3: for i=1..N2, t_index=1..N2, r_index=1..N2 (r!=t), m=1..K2
        for _i in range(N2):  # outer repetition
            for t_rank, t_idx in enumerate(sorted_indices):  # transmitter
                for r_rank, r_idx in enumerate(sorted_indices):  # receiver
                    if r_idx == t_idx:
                        continue
                    for m in range(K2):  # arm index
                        if self.t >= self.T:
                            return

                        arm = available_arms[m]
                        transmitter = idx_to_player[t_idx]
                        receiver = idx_to_player[r_idx]

                        assignment: Dict[int, Optional[int]] = {}

                        # Committed players continue
                        for p, a in committed.items():
                            assignment[p] = a
                            self._do_pull(p, a, update_player=False)

                        # Determine who pulls the communication arm
                        comm_arm_pullers = []

                        # Transmitter pulls comm arm only if Success=0
                        if success.get(transmitter, 0) == 0:
                            comm_arm_pullers.append(transmitter)

                        # Receiver always pulls comm arm
                        comm_arm_pullers.append(receiver)

                        # Resolve conflict on comm arm
                        if len(comm_arm_pullers) > 1:
                            winner = self._arm_choose(arm, comm_arm_pullers)
                            for p in comm_arm_pullers:
                                if p == winner:
                                    assignment[p] = arm
                                    self._do_pull(p, arm, update_player=False)
                                else:
                                    assignment[p] = None  # rejected
                                    if p == receiver:
                                        success[p] = 0
                        elif len(comm_arm_pullers) == 1:
                            p = comm_arm_pullers[0]
                            assignment[p] = arm
                            self._do_pull(p, arm, update_player=False)

                        # Other active players pull arbitrary arms != comm arm
                        used = {arm}
                        for p in active_players:
                            if p in assignment:
                                continue
                            for other_arm in available_arms:
                                if other_arm not in used:
                                    assignment[p] = other_arm
                                    used.add(other_arm)
                                    self._do_pull(p, other_arm, update_player=False)
                                    break
                            else:
                                assignment[p] = None

                        full_asgn = {pp: assignment.get(pp) for pp in range(self.N)}
                        if not self._record_step(full_asgn):
                            return

    def _do_gs_phase(
        self,
        successful_players: List[int],
        available_arms: List[int],
        committed: Dict[int, int],
        all_active: List[int],
    ) -> Dict[int, int]:
        """Execute GS matching (Algorithm 8) for successful players.
        Uses empirical preferences. Costs N2^2 time steps."""
        if not successful_players:
            return {}

        N2 = len(all_active)
        est_p_ranks = _stable_argsort_desc_rows(self.mu_hat_p)
        est_a_ranks = _stable_argsort_desc_rows(self.mu_hat_a)
        gs_result = gale_shapley_player_proposing(
            self.N, self.K, est_p_ranks, est_a_ranks,
            successful_players, list(available_arms),
        )

        # The GS execution takes N2^2 time steps in the paper
        for _ in range(N2 * N2):
            if self.t >= self.T:
                break
            assignment: Dict[int, Optional[int]] = {}
            for p, a in committed.items():
                assignment[p] = a
                self._do_pull(p, a, update_player=False)
            for p in successful_players:
                if p in gs_result:
                    a = gs_result[p]
                    assignment[p] = a
                    self._do_pull(p, a, update_player=False)
            full_asgn = {pp: assignment.get(pp) for pp in range(self.N)}
            if not self._record_step(full_asgn):
                break

        return gs_result

    def _do_availability_check(
        self,
        unsuccessful: List[int],
        available_arms: List[int],
        player_indices: Dict[int, int],
        committed: Dict[int, int],
    ) -> List[int]:
        """Update sub-phase: unsuccessful players check arm availability.
        Algorithm 1, Lines 9-16. Costs N2*K2 time steps.
        Returns updated available_arms list."""
        N2 = len(unsuccessful)
        K2 = len(available_arms)
        if N2 == 0 or K2 == 0:
            return list(available_arms)

        idx_to_player = {}
        for p in unsuccessful:
            idx_to_player[player_indices[p]] = p
        sorted_indices = sorted(idx_to_player.keys())

        new_available = list(available_arms)
        new_N2 = N2

        # for t = 1, ..., N2*K2: player with index n checks m-th arm
        for t in range(1, N2 * K2 + 1):
            if self.t >= self.T:
                break
            # t = (n-1)*K2 + m, so n = (t-1)//K2 + 1, m = (t-1)%K2 + 1
            n_rank = (t - 1) // K2  # 0-based rank
            m_pos = (t - 1) % K2    # 0-based arm position

            if n_rank >= len(sorted_indices) or m_pos >= len(available_arms):
                # Just record a step with committed matching
                assignment = {p: committed.get(p) for p in range(self.N)}
                if not self._record_step(assignment):
                    break
                continue

            idx = sorted_indices[n_rank]
            player = idx_to_player[idx]
            arm = available_arms[m_pos]

            assignment: Dict[int, Optional[int]] = {}
            for p, a in committed.items():
                assignment[p] = a
                self._do_pull(p, a, update_player=False)

            # Player pulls arm to check availability
            if arm in [committed.get(p) for p in committed]:
                # Arm is occupied by a committed player
                # The committed player wins the conflict -> checking player rejected
                assignment[player] = None
                if arm in new_available:
                    new_available.remove(arm)
                    new_N2 -= 1
            else:
                assignment[player] = arm
                self._do_pull(player, arm, update_player=False)

            full_asgn = {pp: assignment.get(pp) for pp in range(self.N)}
            if not self._record_step(full_asgn):
                break

        return new_available

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        committed: Dict[int, int] = {}
        active_players = list(range(self.N))
        available_arms = list(range(self.K))

        # --- Phase 1: Index Assignment ---
        player_indices = self._index_assignment(
            active_players, available_arms, committed
        )

        # --- Phase 2: Round Robin ---
        round_num = 0
        while self.t < self.T and active_players and available_arms:
            round_num += 1
            if self.verbose:
                print(
                    f"Round {round_num}: N2={len(active_players)}, "
                    f"K2={len(available_arms)}, t={self.t}"
                )

            # Sub-phase 1: Exploration (Algorithm 2)
            success = self._do_exploration(
                active_players, available_arms, player_indices, committed
            )
            if self.t >= self.T:
                break

            # Sub-phase 2: Communication (Algorithm 3 COMM)
            self._do_communication(
                active_players, available_arms, player_indices, committed, success
            )
            if self.t >= self.T:
                break

            # Sub-phase 3: Update
            successful = [p for p in active_players if success.get(p, 0) == 1]
            unsuccessful = [p for p in active_players if success.get(p, 0) == 0]

            # GS matching for successful players
            if successful:
                gs_result = self._do_gs_phase(
                    successful, available_arms, committed, active_players
                )
                if self.t >= self.T:
                    break

                # Commit successful players
                for p in successful:
                    if p in gs_result:
                        committed[p] = gs_result[p]
                        if p in active_players:
                            active_players.remove(p)
                        arm = gs_result[p]
                        if arm in available_arms:
                            available_arms.remove(arm)

            if not active_players or not available_arms:
                break

            # Availability check for unsuccessful players
            available_arms = self._do_availability_check(
                unsuccessful, available_arms, player_indices, committed
            )
            if self.t >= self.T:
                break

            # Record epoch boundary
            self.epoch_boundaries.append(self.t)

            # Re-index remaining players
            active_players = [p for p in unsuccessful if p not in committed]
            if not active_players or not available_arms:
                break
            player_indices = self._index_assignment(
                active_players, available_arms, committed, record=True
            )

        # --- Phase 3: Exploitation ---
        # Regret is constant per step (committed matching doesn't change)
        if self.t < self.T:
            exploit_assignment = {p: committed.get(p) for p in range(self.N)}
            exploit_regret = step_regret_from_matching(
                self.env, self.opt_total_utility, exploit_assignment
            )
            remaining = self.T - self.t
            if remaining > 0:
                steps = np.arange(1, remaining + 1, dtype=np.float64)
                self.cum_regret[self.t:] = self.cum + exploit_regret * steps
                self.cum += exploit_regret * remaining
                self.t = self.T

        return self.cum_regret
