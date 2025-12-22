import matplotlib.pyplot as plt

from src.constants import TUNING_BUDGET, SEED
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    plot_accumulated_timestamp_on_ax,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
MOCK_SEED = 0


def get_results(
    impl: str,
    dataset: str,
    solutions: list,
    recall_min: float = None,
    qps_min: int = None,
    sampling_count: int = None,
    tuning_time: int = TUNING_BUDGET,
):
    results_combi = {}

    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)

        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            results = [(hp, perf)]
        else:
            results = [r for r in results if r[1][0] <= tuning_time]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
    }


def main():
    import matplotlib.font_manager as fm

    # Register font
    font_path = f"{current_dir}/LinLibertine_R.ttf"
    if font_path not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path)

    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    # --- Config ---
    SOLUTIONS = [
        "our_solution",
        "brute_force",   # If your code uses a different key, change it here
        "random_search",
        "grid_search",
        "vd_tuner",
        "optuna",
        "nsga",
        # "brute_force",      # include only if you actually plot it
    ]

    # Legend order you want (labels must match what plot_accumulated_timestamp_on_ax puts in legend)
    LEGEND_ORDER = [
        "CHAT",
        "Oracle Solution",
        "Random Search",
        "Grid Search",
        "VDTuner",
        "Optuna",
        "NSGA-II",
    ]

    IMPL = "milvus"
    DATASET = "nytimes-256-angular"
    SAMPLING_COUNT = 10

    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"  # choose "q50", "q75", "q90", ...
    QPS_MIN = get_qps_metrics_dataset(IMPL, DATASET, ret_dict=True)[QPS_MIN_KEY]

    # Two panels only:
    # (1) recall_min constraint => QPS vs time
    # (2) qps_min constraint => Recall vs time
    results_left = get_results(
        impl=IMPL,
        dataset=DATASET,
        solutions=SOLUTIONS,
        recall_min=RECALL_MIN,
        qps_min=None,
        sampling_count=SAMPLING_COUNT,
        tuning_time=TUNING_BUDGET,
    )
    results_right = get_results(
        impl=IMPL,
        dataset=DATASET,
        solutions=SOLUTIONS,
        recall_min=None,
        qps_min=QPS_MIN,
        sampling_count=SAMPLING_COUNT,
        tuning_time=TUNING_BUDGET,
    )

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    plot_accumulated_timestamp_on_ax(
        axes[0],
        results_left["results"],
        results_left["recall_min"],
        results_left["qps_min"],
    )
    plot_accumulated_timestamp_on_ax(
        axes[1],
        results_right["results"],
        results_right["recall_min"],
        results_right["qps_min"],
    )

    # Optional: panel titles (remove if you don't want them)
    # axes[0].set_title(f"Recall ≥ {RECALL_MIN}", fontsize=16)
    # axes[1].set_title(f"QPS ≥ {QPS_MIN} ({QPS_MIN_KEY})", fontsize=16)

    # --- Legend (deduplicate + enforce order + avoid overflow) ---
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Deduplicate (keep last occurrence)
    by_label = dict(zip(labels, handles))

    # Reorder by LEGEND_ORDER (keep only existing labels)
    ordered_labels = [lab for lab in LEGEND_ORDER if lab in by_label]
    ordered_handles = [by_label[lab] for lab in ordered_labels]

    # Add any leftover labels not in LEGEND_ORDER (optional)
    leftovers = [lab for lab in by_label.keys() if lab not in ordered_labels]
    ordered_labels += leftovers
    ordered_handles += [by_label[lab] for lab in leftovers]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),  # push legend above to prevent clipping
        ncol=min(7, max(1, len(ordered_labels))),
        fontsize=14,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.6,
    )

    # Layout: reserve top space for legend
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    fig.savefig(f"milvus_nytimes_{SEED}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
