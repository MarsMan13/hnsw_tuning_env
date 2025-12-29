import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.utils import (
    filename_builder,
    get_local_optimal_hyperparameter,
    load_search_results,
)

SEED = 42

# Allowed M values (only these will be plotted)
ALLOWED_M = {8, 12, 16, 20, 24, 32, 40, 48, 56, 64}

# efC interpolation range (integer grid)
EFC_MIN, EFC_MAX = 10, 576
EFC_GRID = np.arange(EFC_MIN, EFC_MAX + 1, dtype=int)


def _masked_gaussian_filter1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply 1D Gaussian smoothing while ignoring NaNs.
    NaN positions stay NaN if there is no supporting mass nearby.
    """
    mask = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0.0)

    num = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter1d(mask.astype(np.float32), sigma=sigma, mode="nearest")

    out = np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0))
    return out


def _select_best_per_efc(
    candidates: list[tuple[tuple[int, int, int], tuple]],
    recall_min: float | None,
    qps_min: int | None,
):
    """
    Given candidates sharing same (M, efC) but different efS,
    pick a single representative point consistent with the constraint:
      - Recall constraint: among recall>=recall_min pick max QPS
      - QPS constraint: among qps>=qps_min pick max Recall

    Expects perf like:
      perf = (t, recall, qps, total_time, build_time, index_size)

    Returns:
      (recall, qps, total_time, build_time, index_size) or None if infeasible.
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    best = None
    best_key = None

    for (_hp, perf) in candidates:
        _t, recall, qps, total_time, build_time, index_size = perf

        if recall_min is not None:
            if recall < recall_min:
                continue
            key = float(qps)  # maximize qps under recall constraint
        else:
            if qps < qps_min:
                continue
            key = float(recall)  # maximize recall under qps constraint

        if best is None or key > best_key:
            best = (
                float(recall),
                float(qps),
                float(total_time),
                float(build_time),
                float(index_size),
            )
            best_key = key

    return best


def smooth_fill_results_over_efc(
    results: list[tuple[tuple[int, int, int], tuple]],
    allowed_m: set[int],
    efc_grid: np.ndarray,
    recall_min: float | None,
    qps_min: int | None,
    sigma: float = 3.0,
) -> list[tuple[tuple[int, int, int], tuple]]:
    """
    Build a dense (M, efC) grid by:
      1) collapsing efS for each (M, efC) into one best point (per constraint)
      2) smoothing each metric along efC with masked Gaussian
      3) emitting filled results as ((M, efC, efS_dummy), perf)

    efS is set to 0 for filled points (dummy), since we are smoothing along efC.
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    # Group raw results by (M, efC)
    bucket: dict[tuple[int, int], list[tuple[tuple[int, int, int], tuple]]] = {}
    for hp, perf in results:
        M, efC, efS = hp
        M = int(M)
        efC = int(efC)
        efS = int(efS)

        if M not in allowed_m:
            continue
        if efC < int(efc_grid[0]) or efC > int(efc_grid[-1]):
            continue

        bucket.setdefault((M, efC), []).append(((M, efC, efS), perf))

    filled_results: list[tuple[tuple[int, int, int], tuple]] = []

    for M in sorted(allowed_m):
        # Prepare per-efC arrays (NaN where missing/infeasible)
        recall_arr = np.full(len(efc_grid), np.nan, dtype=np.float64)
        qps_arr = np.full(len(efc_grid), np.nan, dtype=np.float64)
        total_time_arr = np.full(len(efc_grid), np.nan, dtype=np.float64)
        build_time_arr = np.full(len(efc_grid), np.nan, dtype=np.float64)
        index_size_arr = np.full(len(efc_grid), np.nan, dtype=np.float64)

        for i, efC in enumerate(efc_grid.tolist()):
            cand = bucket.get((M, int(efC)), None)
            if not cand:
                continue

            best = _select_best_per_efc(cand, recall_min=recall_min, qps_min=qps_min)
            if best is None:
                continue

            recall, qps, total_time, build_time, index_size = best
            recall_arr[i] = recall
            qps_arr[i] = qps
            total_time_arr[i] = total_time
            build_time_arr[i] = build_time
            index_size_arr[i] = index_size

        # Smooth along efC
        recall_s = _masked_gaussian_filter1d(recall_arr, sigma=sigma)
        qps_s = _masked_gaussian_filter1d(qps_arr, sigma=sigma)
        total_time_s = _masked_gaussian_filter1d(total_time_arr, sigma=sigma)
        build_time_s = _masked_gaussian_filter1d(build_time_arr, sigma=sigma)
        index_size_s = _masked_gaussian_filter1d(index_size_arr, sigma=sigma)

        # Emit dense points; keep only positions that are still valid (not NaN)
        for i, efC in enumerate(efc_grid.tolist()):
            if np.isnan(recall_s[i]) or np.isnan(qps_s[i]):
                continue

            hp = (int(M), int(efC), 0)  # efS dummy
            perf = (
                0.0,
                float(recall_s[i]),
                float(qps_s[i]),
                float(total_time_s[i]) if not np.isnan(total_time_s[i]) else 0.0,
                float(build_time_s[i]) if not np.isnan(build_time_s[i]) else 0.0,
                int(index_size_s[i]) if not np.isnan(index_size_s[i]) else 0,
            )
            filled_results.append((hp, perf))

    return filled_results


def load_local_optima_by_M(
    impl: str,
    dataset: str,
    solution: str,
    recall_min: float = None,
    qps_min: int = None,
    sampling_count: int = 10,
    sigma: float = 3.0,
):
    """
    Load raw results, smooth-fill efC grid per allowed M, and compute local optima.
    Returns arrays for plotting: x=efC, y=M
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
    results = load_search_results(solution, filename, seed=SEED, sampling_count=sampling_count)

    # 1) Smooth-fill results over efC for each allowed M
    filled_results = smooth_fill_results_over_efc(
        results=results,
        allowed_m=ALLOWED_M,
        efc_grid=EFC_GRID,
        recall_min=recall_min,
        qps_min=qps_min,
        sigma=sigma,
    )

    # 2) Compute local optima on the dense, smoothed grid
    local_opt_list = get_local_optimal_hyperparameter(filled_results, recall_min, qps_min)

    # Keep only allowed M and return (x=efC, y=M)
    pairs = []
    for hp, _perf in local_opt_list:
        M, efC, _efS = hp
        M = int(M)
        efC = int(efC)
        if M in ALLOWED_M:
            pairs.append((M, efC))

    pairs.sort(key=lambda x: x[0])  # sort by M
    efc_vals = [p[1] for p in pairs]
    m_vals = [p[0] for p in pairs]
    return efc_vals, m_vals


def plot_two_constraints_side_by_side(
    impl: str,
    datasets: list[str],
    solution: str,
    recall_min: float,
    qps_min: int,
    sampling_count: int = 10,
    sigma: float = 3.0,
    out_path: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: Recall constraint
    ax = axes[0]
    for dataset in datasets:
        efc_vals, m_vals = load_local_optima_by_M(
            impl=impl,
            dataset=dataset,
            solution=solution,
            recall_min=recall_min,
            qps_min=None,
            sampling_count=sampling_count,
            sigma=sigma,
        )
        ax.plot(efc_vals, m_vals, marker="o", linewidth=2, label=dataset)

    ax.set_title(f"Recall constraint (Recall ≥ {recall_min})")
    ax.set_xlabel("efC")
    ax.set_ylabel("M")
    ax.set_yticks(sorted(ALLOWED_M))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Right: QPS constraint
    ax = axes[1]
    for dataset in datasets:
        efc_vals, m_vals = load_local_optima_by_M(
            impl=impl,
            dataset=dataset,
            solution=solution,
            recall_min=None,
            qps_min=qps_min,
            sampling_count=sampling_count,
            sigma=sigma,
        )
        ax.plot(efc_vals, m_vals, marker="o", linewidth=2, label=dataset)

    ax.set_title(f"QPS constraint (QPS ≥ {qps_min})")
    ax.set_xlabel("efC")
    ax.set_yticks(sorted(ALLOWED_M))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if out_path:
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()


def main():
    RECALL_MIN = 0.95
    QPS_MIN = 100000

    DATASETS = [
        "nytimes-256-angular-100p",
        "nytimes-256-angular-50p",
        "nytimes-256-angular-10p",
        "nytimes-256-angular-5p",
    ]

    plot_two_constraints_side_by_side(
        impl="faiss",
        datasets=DATASETS,
        solution="brute_force",
        recall_min=RECALL_MIN,
        qps_min=QPS_MIN,
        sampling_count=10,
        sigma=3.0,
        out_path="local_optima_efC_M_two_constraints_smoothed.png",
    )


if __name__ == "__main__":
    main()
