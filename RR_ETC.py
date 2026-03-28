import math
from typing import Dict, List, Optional

import numpy as np

from environment import (
    compute_player_optimal_matching,
    matching_total_utility,
    step_regret_from_matching,
    _stable_argsort_desc_rows,
    gale_shapley_player_proposing,
)


class RRETC_Full:
    """
    Faithful implementation of Algorithm 1 (Round-Robin ETC) from
    Zhang & Fang, UAI 2024. Implements all phases exactly as defined:
      Phase 1: Index Assignment
      Phase 2: Round Robin (Exploration + Communication + Update)
      Phase 3: Exploitation
      
    - Arms use an empirical-leader rational logic.
    - Players only update statistics during the exploration phase.
    - Arms constantly update statistics across the entire horizon.
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
        """Arm intelligently selects the player with the highest empirical mean."""
        if not candidates:
            return -1
        
        best = candidates[0]
        best_val = self.mu_hat_a[arm, best]
        for p in candidates[1:]:
            v = self.mu_hat_a[arm, p]
            # Tie breaker: favor smaller ID
            if v > best_val or (v == best_val and p < best):
                best, best_val = p, v
        return best

    def _record_step(self, assignment: Dict[int, Optional[int]]) -> bool:
        """Processes a single step, logs regret, and updates horizon checks."""
        if self.t >= self.T:
            return False
            
        sr = step_regret_from_matching(self.env, self.opt_total_utility, assignment)
        self.cum += sr
        self.cum_regret[self.t] = self.cum
        self.t += 1
        return True

    def _do_pull(self, player: int, arm: int, update_player: bool = True):
        """Executes a single pull logic and updates internal stats arrays."""
        r_p, r_a = self.env.step(player, arm)
        
        if update_player:
            n = self.n_pulls_p[player, arm]
            self.mu_hat_p[player, arm] = (self.mu_hat_p[player, arm] * n + r_p) / (n + 1)
            self.n_pulls_p[player, arm] = n + 1
            
        n_a = self.n_pulls_a[arm, player]
        self.mu_hat_a[arm, player] = (self.mu_hat_a[arm, player] * n_a + r_a) / (n_a + 1)
        self.n_pulls_a[arm, player] = n_a + 1

    def _has_confident_estimation(self, player: int, arm_list: List[int]) -> bool:
        """
        Algorithm 2 Line 4 logic check.
        Ensures that all available arm pairs for a given player can be statistically separated 
        via confidence bounds.
        """
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
                
                # Check overlapping logic
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

    def run(self) -> np.ndarray:
        """
        Executes the three main operational phases up to given T limit.
        """
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
                
        # Phase 3: Exploitation (Vectorized processing for final horizon offset)
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
