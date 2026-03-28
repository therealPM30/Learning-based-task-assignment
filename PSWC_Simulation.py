import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Import environments and helpers
from environment import (
    PaperEnvironment,
    MRTAEnvironment,
    RANDOM_SEED,
    setup_appendix_e_case1_exact,
    setup_appendix_e_case2_exact,
    setup_appendix_e_case3_exact,
)

# Import algorithms
from RR_ETC import RRETC_Full
from CA_ETC import CA_ETC_Matching
from PCA_UCB import PCA_UCB_Matching


# ================================================================
# Appendix E / Benchmark Runners and Random Experiments
# ================================================================

def _clone_env_for_seed(src_env):
    """Deep copy utilities so each algorithm evaluates using identical means."""
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
    Run Appendix E benchmarks (Cases 1-3) with RR-ETC, PCA-UCB, and CA-ETC.
    
    Correction alignment requirements enforced:
    - Horizon up to 5,000,000
    - Generates average values across provided seeds and plots the respective means
    - Custom plot scaling limits
    - Epoch round boundaries rendered as vertical markers for RR-ETC
    """

    def make_env_for_case(case_id: int) -> PaperEnvironment:
        if case_id == 1:
            return setup_appendix_e_case1_exact()
        if case_id == 2:
            return setup_appendix_e_case2_exact()
        if case_id == 3:
            return setup_appendix_e_case3_exact()
        raise ValueError("case_id must be 1, 2, or 3")

    # Clean matplotlib visual baseline standardizations
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5
    })

    colors = {
        'pca': '#2ca02c', # Green
        'ca': '#1f77b4',  # Blue
        'rr': '#d62728'   # Red
    }

    # Restricting view specifically onto Case 1 to avoid clutter
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    case_id = 1

    # Bookkeeping structures
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

        # Initialize configurations mapping specific tracking parameters mapping
        rr = RRETC_Full(env_rr, horizon=T, c_confidence=c_confidence, verbose=False)
        pca = PCA_UCB_Matching(env_pca, horizon=T, phase_length=100,
                               c_confidence=c_confidence, lambda_delay=0.87)
        ca = CA_ETC_Matching(env_ca, horizon=T, T0=130000, gamma=0.35,
                             epoch_horizon_growth=7.0)

        # Run independent simulation logic configurations
        all_rr[seed] = rr.run()
        all_pca[seed] = pca.run()
        all_ca[seed] = ca.run()

        if epoch_bounds_reference is None:
            epoch_bounds_reference = rr.epoch_boundaries

    # Calculate overarching stats tracking metrics
    mean_rr = all_rr.mean(axis=0)
    mean_pca = all_pca.mean(axis=0)
    mean_ca = all_ca.mean(axis=0)
    std_rr = all_rr.std(axis=0)
    std_pca = all_pca.std(axis=0)
    std_ca = all_ca.std(axis=0)

    ts = np.arange(1, T + 1)

    # Plot baseline constraints tracking requirements
    ax.set_xticks(np.arange(0, 60_001, 6_000))
    ax.set_yticks(np.arange(0, 150_001, 15_000))
    ax.set_xlim(0, 60_000)
    ax.set_ylim(0, 150_000)

    ax.set_xlabel("Time (Steps)")
    ax.set_ylabel("Cumulative Player Regret")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # -------------------------------------------------------------------------
    # Uncomment lines below to display specific cases concurrently:
    # -------------------------------------------------------------------------

    # # 1. Case: Only PCA-UCB
    # ax.plot(ts, mean_pca, color=colors['pca'], label="PCA-UCB")
    # if num_seeds > 1:
    #     ax.fill_between(ts, mean_pca - std_pca, mean_pca + std_pca, color=colors['pca'], alpha=0.15)

    # # 2. Case: CA-ETC (Uncomment PCA-UCB as well to combine visualizations)
    # ax.plot(ts, mean_ca, color=colors['ca'], label="CA-ETC")
    # if num_seeds > 1:
    #     ax.fill_between(ts, mean_ca - std_ca, mean_ca + std_ca, color=colors['ca'], alpha=0.15)

    # 3. Case: RR-ETC (Uncomment all three to generate a complete visual configuration benchmark)
    ax.plot(ts, mean_rr, color=colors['rr'], label="Round-Robin ETC")
    if num_seeds > 1:
        ax.fill_between(ts, mean_rr - std_rr, mean_rr + std_rr, color=colors['rr'], alpha=0.15)

    # Optional marker implementation highlighting Epoch separation logic steps
    # if epoch_bounds_reference is not None:
    #     for eb in epoch_bounds_reference:
    #         ax.axvline(eb, color="k", linestyle=":", alpha=0.3)

    # -------------------------------------------------------------------------

    ax.legend(loc="upper left", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Full evaluation entry point:
      - Plot 1: Standard benchmark rendering corresponding matching algorithms across Cases 1-3.
    """
    run_appendix_e_benchmarks(T=60_000, num_seeds=1, c_confidence=1.41421356237)


if __name__ == "__main__":
    main()
