import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from src.utils import (
    filename_builder,
    get_local_optimal_hyperparameter,
    load_search_results,
)

SEED = 42

# Allowed M values (only these will be plotted)
ALLOWED_M = {8, 12, 16, 20, 24, 32, 40, 48, 56, 64}
ALLOWED_M = {i for i in range(16, 65, 4)}  # 8,12,...,64

# efC interpolation range (integer grid)
EFC_MIN, EFC_MAX = 10, 576
EFC_GRID = np.arange(EFC_MIN, EFC_MAX + 1, dtype=int)

def masked_gaussian_filter2d(matrix: np.ndarray, sigma_m: float, sigma_efc: float) -> np.ndarray:
    """
    2D Gaussian smoothing while ignoring NaNs.
    matrix: shape (n_M, n_efC), NaN indicates missing.
    """
    assert matrix.ndim == 2

    mask = ~np.isnan(matrix)
    filled = np.nan_to_num(matrix, nan=0.0)

    num = gaussian_filter(filled, sigma=(sigma_m, sigma_efc), mode="nearest")
    den = gaussian_filter(mask.astype(np.float32), sigma=(sigma_m, sigma_efc), mode="nearest")

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

def smooth_fill_results_over_M_efC_2d(
    results: list[tuple[tuple[int, int, int], tuple]],
    allowed_m: list[int],
    efc_grid: np.ndarray,
    recall_min: float | None,
    qps_min: int | None,
    sigma_m: float = 1.0,
    sigma_efc: float = 3.0,
) -> list[tuple[tuple[int, int, int], tuple]]:
    """
    2D smoothing over (M, efC) grid.
    Returns filled_results as list of ((M, efC, efS_dummy), perf).
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    allowed_m_sorted = sorted(allowed_m)
    m_to_i = {m: i for i, m in enumerate(allowed_m_sorted)}
    efc_to_j = {int(e): j for j, e in enumerate(efc_grid.tolist())}

    nM = len(allowed_m_sorted)
    nE = len(efc_grid)

    # Step 1) group by (M, efC) and keep best over efS
    bucket: dict[tuple[int, int], list[tuple[tuple[int, int, int], tuple]]] = {}
    for hp, perf in results:
        M, efC, efS = map(int, hp)
        if M not in m_to_i:
            continue
        if efC not in efc_to_j:
            continue
        bucket.setdefault((M, efC), []).append(((M, efC, efS), perf))

    # Allocate metric grids
    recall_mat = np.full((nM, nE), np.nan, dtype=np.float64)
    qps_mat = np.full((nM, nE), np.nan, dtype=np.float64)
    total_time_mat = np.full((nM, nE), np.nan, dtype=np.float64)
    build_time_mat = np.full((nM, nE), np.nan, dtype=np.float64)
    index_size_mat = np.full((nM, nE), np.nan, dtype=np.float64)

    # Reuse your _select_best_per_efc (unchanged)
    for (M, efC), cand in bucket.items():
        best = _select_best_per_efc(cand, recall_min=recall_min, qps_min=qps_min)
        if best is None:
            continue
        recall, qps, total_time, build_time, index_size = best
        i = m_to_i[M]
        j = efc_to_j[efC]
        recall_mat[i, j] = recall
        qps_mat[i, j] = qps
        total_time_mat[i, j] = total_time
        build_time_mat[i, j] = build_time
        index_size_mat[i, j] = index_size

    # Step 2) 2D masked smoothing
    recall_s = masked_gaussian_filter2d(recall_mat, sigma_m=sigma_m, sigma_efc=sigma_efc)
    qps_s = masked_gaussian_filter2d(qps_mat, sigma_m=sigma_m, sigma_efc=sigma_efc)
    total_time_s = masked_gaussian_filter2d(total_time_mat, sigma_m=sigma_m, sigma_efc=sigma_efc)
    build_time_s = masked_gaussian_filter2d(build_time_mat, sigma_m=sigma_m, sigma_efc=sigma_efc)
    index_size_s = masked_gaussian_filter2d(index_size_mat, sigma_m=sigma_m, sigma_efc=sigma_efc)

    # Step 3) emit filled results
    filled_results: list[tuple[tuple[int, int, int], tuple]] = []
    for i, M in enumerate(allowed_m_sorted):
        for j, efC in enumerate(efc_grid.tolist()):
            if np.isnan(recall_s[i, j]) or np.isnan(qps_s[i, j]):
                continue

            hp = (int(M), int(efC), 0)  # efS dummy
            perf = (
                0.0,
                float(recall_s[i, j]),
                float(qps_s[i, j]),
                float(total_time_s[i, j]) if not np.isnan(total_time_s[i, j]) else 0.0,
                float(build_time_s[i, j]) if not np.isnan(build_time_s[i, j]) else 0.0,
                int(index_size_s[i, j]) if not np.isnan(index_size_s[i, j]) else 0,
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
    sigma_m: float = 1.0,
    sigma_efc: float = 3.0,
):
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
    results = load_search_results(solution, filename, seed=SEED, sampling_count=sampling_count)

    filled_results = smooth_fill_results_over_M_efC_2d(
        results=results,
        allowed_m=sorted(ALLOWED_M),
        efc_grid=EFC_GRID,
        recall_min=recall_min,
        qps_min=qps_min,
        sigma_m=sigma_m,
        sigma_efc=sigma_efc,
    )

    local_opt_list = get_local_optimal_hyperparameter(filled_results, recall_min, qps_min)

    pairs = []
    for hp, _perf in local_opt_list:
        M, efC, _ = map(int, hp)
        if M in ALLOWED_M:
            pairs.append((M, efC))

    pairs.sort(key=lambda x: x[0])
    efc_vals = [p[1] for p in pairs]  # x
    m_vals = [p[0] for p in pairs]    # y
    return efc_vals, m_vals

def plot_two_constraints_side_by_side(
    impl: str,
    datasets: list[str],
    solution: str,
    recall_min: float,
    qps_min: int,
    sampling_count: int = 10,
    sigma_m_left: float = 1.0,
    sigma_efc_left: float = 4.0,
    sigma_m_right: float = 1.0,
    sigma_efc_right: float = 4.0,
    out_path: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # ---------------------------
    # Left: Recall constraint
    # ---------------------------
    ax = axes[0]
    for dataset in datasets:
        efc_vals, m_vals = load_local_optima_by_M(
            impl=impl,
            dataset=dataset,
            solution=solution,
            recall_min=recall_min,
            qps_min=None,
            sampling_count=sampling_count,
            sigma_m=sigma_m_left,
            sigma_efc=sigma_efc_left,
        )
        ax.plot(efc_vals, m_vals, marker="o", linewidth=2, label=dataset)

    ax.set_title(f"Recall constraint (Recall ≥ {recall_min})")
    ax.set_xlabel("efC")
    ax.set_ylabel("M")
    ax.set_yticks(sorted(ALLOWED_M))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # ---------------------------
    # Right: QPS constraint
    # ---------------------------
    ax = axes[1]
    for dataset in datasets:
        efc_vals, m_vals = load_local_optima_by_M(
            impl=impl,
            dataset=dataset,
            solution=solution,
            recall_min=None,
            qps_min=qps_min,
            sampling_count=sampling_count,
            sigma_m=sigma_m_right,
            sigma_efc=sigma_efc_right,
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
    RECALL_MIN = 0.90
    QPS_MIN = 100000

    # faiss
    DATASETS = [
        "nytimes-256-angular-100p",
        # "nytimes-256-angular-50p",
        "nytimes-256-angular-10p",
        # "nytimes-256-angular-5p",
    ]

    plot_two_constraints_side_by_side(
        impl="faiss",
        datasets=DATASETS,
        solution="brute_force",
        recall_min=RECALL_MIN,
        qps_min=QPS_MIN,
        sampling_count=10,
        sigma_m_left=2.2,
        sigma_efc_left=4.7,
        sigma_m_right=0.9,
        sigma_efc_right=0.1,
        out_path="local_optima_efC_M_two_constraints_smoothed.png",
    )

if __name__ == "__main__":
    main()
