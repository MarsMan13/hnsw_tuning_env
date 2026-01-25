import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    filename_builder,
    load_search_results,
)

SEED = 42

# Allowed M values (only these will be plotted)
ALLOWED_M = list(range(8, 65, 8))  # 8, 16, ..., 64

# If your result file already contains all efC values (dense grid), you don't need EFC_GRID.
# Keep it only if you want to enforce a plotting range.
EFC_MIN, EFC_MAX = 10, 1024


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
            key = float(qps)  # maximize QPS
        else:
            if float(qps) < float(qps_min):
                continue
            key = float(recall)  # maximize Recall

        if best is None or key > best_key:
            best = (float(recall), float(qps), float(total_time), float(build_time), float(index_size))
            best_key = key

    return best


def pick_best_efc_per_M_no_smoothing(
    results: list[tuple[tuple[int, int, int], tuple]],
    allowed_m: list[int],
    recall_min: float | None,
    qps_min: int | None,
    efc_min: int = EFC_MIN,
    efc_max: int = EFC_MAX,
) -> dict[int, int]:
    """
    Assumes the result file already contains smoothed/interpolated performance
    for each (M, efC) on a dense efC grid.

    For each fixed M:
      - filter by constraint (Recall>=recall_min OR QPS>=qps_min)
      - pick efC that maximizes the objective (QPS or Recall)

    Returns:
      dict: M -> best_efC
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    allowed_m_sorted = sorted(allowed_m)
    allowed_m_set = set(allowed_m_sorted)

    # Bucket by (M, efC) because there can still be multiple efS per efC
    bucket: dict[tuple[int, int], list[tuple[tuple[int, int, int], tuple]]] = {}
    for hp, perf in results:
        M, efC, efS = map(int, hp)
        if M not in allowed_m_set:
            continue
        if efC < efc_min or efC > efc_max:
            continue
        bucket.setdefault((M, efC), []).append(((M, efC, efS), perf))

    out: dict[int, int] = {}

    for M in allowed_m_sorted:
        # Iterate only efC values observed for this M
        efcs = sorted([efC for (m_key, efC) in bucket.keys() if m_key == M])
        if not efcs:
            continue

        best_efc = None
        best_obj = None

        for efC in efcs:
            best = _select_best_per_efc(bucket[(M, efC)], recall_min=recall_min, qps_min=qps_min)
            if best is None:
                continue

            recall, qps, _total_time, _build_time, _index_size = best
            obj = float(qps) if recall_min is not None else float(recall)

            if best_obj is None or obj > best_obj:
                best_obj = obj
                best_efc = efC

        if best_efc is not None:
            out[M] = int(best_efc)

    return out


def load_local_optima_by_M(
    impl: str,
    dataset: str,
    solution: str,
    recall_min: float | None = None,
    qps_min: int | None = None,
    sampling_count: int = 10,
):
    """
    Returns:
      efc_vals (x), m_vals (y)
    efc_vals are selected directly from the already-smoothed/interpolated results.
    """
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
    results = load_search_results(solution, filename, seed=SEED, sampling_count=sampling_count)

    best_map = pick_best_efc_per_M_no_smoothing(
        results=results,
        allowed_m=ALLOWED_M,
        recall_min=recall_min,
        qps_min=qps_min,
        efc_min=EFC_MIN,
        efc_max=EFC_MAX,
    )

    pairs = [(M, best_map[M]) for M in ALLOWED_M if M in best_map]
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
        )
        ax.plot(efc_vals, m_vals, marker="o", linewidth=2, label=dataset)

    ax.set_title(f"Recall constraint (Recall ≥ {recall_min})")
    ax.set_xlabel("efC")
    ax.set_ylabel("M")
    ax.set_yticks(ALLOWED_M)
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
        )
        ax.plot(efc_vals, m_vals, marker="o", linewidth=2, label=dataset)

    ax.set_title(f"QPS constraint (QPS ≥ {qps_min})")
    ax.set_xlabel("efC")
    ax.set_yticks(ALLOWED_M)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if out_path:
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()


def main():
    RECALL_MIN = 0.95
    QPS_MIN = 110000

    DATASETS = [
        "synthetic-128-angular-100p",
        "synthetic-128-angular-10p",
        "synthetic-128-angular-1p",
    ]

    plot_two_constraints_side_by_side(
        impl="faiss",
        datasets=DATASETS,
        solution="brute_force",
        recall_min=RECALL_MIN,
        qps_min=QPS_MIN,
        sampling_count=10,
        out_path="local_optima_efC_M.png",
    )


if __name__ == "__main__":
    main()
