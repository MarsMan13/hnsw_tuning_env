import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.utils import (
    filename_builder,
    load_search_results,
)

SEED = 42

# Allowed M values (only these will be plotted)
ALLOWED_M = {i for i in range(8, 65, 4)}  # 8,12,...,64

# efC grid for 1D curve building (dense integer grid)
EFC_MIN, EFC_MAX = 10, 1024
EFC_GRID = np.arange(EFC_MIN, EFC_MAX + 1, dtype=int)


def _select_best_per_efc(
    candidates: list[tuple[tuple[int, int, int], tuple]],
    recall_min: float | None,
    qps_min: int | None,
):
    """
    For a fixed (M, efC), there can be multiple efS points.
    Pick a single representative point consistent with the constraint:
      - Recall constraint: among recall >= recall_min, pick max QPS
      - QPS constraint: among qps >= qps_min, pick max Recall

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
            if float(recall) < float(recall_min):
                continue
            key = float(qps)  # maximize qps
        else:
            if float(qps) < float(qps_min):
                continue
            key = float(recall)  # maximize recall

        if best is None or key > best_key:
            best = (float(recall), float(qps), float(total_time), float(build_time), float(index_size))
            best_key = key

    return best


def masked_gaussian_filter1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """
    1D Gaussian smoothing while ignoring NaNs.
    values: shape (n,), NaN indicates missing.
    """
    assert values.ndim == 1

    mask = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0.0)

    num = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter1d(mask.astype(np.float32), sigma=sigma, mode="nearest")

    out = np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0))
    return out


def pick_best_efc_per_M_after_smoothing(
    results: list[tuple[tuple[int, int, int], tuple]],
    allowed_m: list[int],
    efc_grid: np.ndarray,
    recall_min: float | None,
    qps_min: int | None,
    sigma_efc: float = 30.0,
):
    """
    Implements:
      1) Fix each M
      2) Build objective curve over efC (using observed points, after collapsing efS)
      3) 1D-smooth the curve along efC
      4) Pick efC that maximizes the smoothed objective for that M

    Returns:
      dict: M -> best_efC
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    allowed_m_sorted = sorted(allowed_m)
    allowed_m_set = set(allowed_m_sorted)
    efc_to_j = {int(e): j for j, e in enumerate(efc_grid.tolist())}

    # Bucket candidates by (M, efC) to resolve efS multiplicity
    bucket: dict[tuple[int, int], list[tuple[tuple[int, int, int], tuple]]] = {}
    for hp, perf in results:
        M, efC, efS = map(int, hp)
        if M not in allowed_m_set:
            continue
        if int(efC) not in efc_to_j:
            continue
        bucket.setdefault((M, int(efC)), []).append(((M, int(efC), int(efS)), perf))

    out: dict[int, int] = {}

    for M in allowed_m_sorted:
        # Step 2: objective curve over efC (NaN where missing/unfeasible)
        obj = np.full((len(efc_grid),), np.nan, dtype=np.float64)

        for efC in efc_grid.tolist():
            key = (int(M), int(efC))
            if key not in bucket:
                continue

            best = _select_best_per_efc(bucket[key], recall_min=recall_min, qps_min=qps_min)
            if best is None:
                continue

            recall, qps, _total_time, _build_time, _index_size = best
            val = float(qps) if recall_min is not None else float(recall)
            obj[efc_to_j[int(efC)]] = val

        if np.all(np.isnan(obj)):
            continue

        # Step 3: 1D smoothing along efC
        obj_s = masked_gaussian_filter1d(obj, sigma=float(sigma_efc))
        if np.all(np.isnan(obj_s)):
            continue

        # Step 4: pick argmax efC on the smoothed curve
        j_star = int(np.nanargmax(obj_s))
        out[int(M)] = int(efc_grid[j_star])

    return out


def load_local_optima_by_M(
    impl: str,
    dataset: str,
    solution: str,
    recall_min: float | None = None,
    qps_min: int | None = None,
    sampling_count: int = 10,
    sigma_efc: float = 6.0,
):
    """
    Returns:
      efc_vals (x), m_vals (y)
    efc_vals are chosen after 1D smoothing along efC for each fixed M.
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
    results = load_search_results(solution, filename, seed=SEED, sampling_count=sampling_count)

    best_map = pick_best_efc_per_M_after_smoothing(
        results=results,
        allowed_m=sorted(ALLOWED_M),
        efc_grid=EFC_GRID,
        recall_min=recall_min,
        qps_min=qps_min,
        sigma_efc=sigma_efc,
    )

    pairs = [(M, best_map[M]) for M in sorted(ALLOWED_M) if M in best_map]
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
    sigma_efc_left: float = 6.0,
    sigma_efc_right: float = 6.0,
    out_path: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: Recall constraint (maximize QPS)
    ax = axes[0]
    for dataset in datasets:
        efc_vals, m_vals = load_local_optima_by_M(
            impl=impl,
            dataset=dataset,
            solution=solution,
            recall_min=recall_min,
            qps_min=None,
            sampling_count=sampling_count,
            sigma_efc=sigma_efc_left,
        )
        ax.plot(efc_vals, m_vals, marker="o", linewidth=2, label=dataset)

    ax.set_title(f"Recall constraint (Recall ≥ {recall_min})")
    ax.set_xlabel("efC")
    ax.set_ylabel("M")
    ax.set_yticks(sorted(ALLOWED_M))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Right: QPS constraint (maximize Recall)
    ax = axes[1]
    for dataset in datasets:
        efc_vals, m_vals = load_local_optima_by_M(
            impl=impl,
            dataset=dataset,
            solution=solution,
            recall_min=None,
            qps_min=qps_min,
            sampling_count=sampling_count,
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
    RECALL_MIN = 0.925
    QPS_MIN = 60000

    DATASETS = [
        "nytimes-256-angular-100p-hnswlib-random",
        "nytimes-256-angular-10p-hnswlib-random",
        "nytimes-256-angular-1p-hnswlib-random",
    ]

    plot_two_constraints_side_by_side(
        impl="hnswlib",
        datasets=DATASETS,
        solution="brute_force",
        recall_min=RECALL_MIN,
        qps_min=QPS_MIN,
        sampling_count=10,
        sigma_efc_left=8.0,
        sigma_efc_right=8.0,
        out_path="local_optima_efC_M_two_constraints_smoothed_hnswlib.png",
    )


if __name__ == "__main__":
    main()
