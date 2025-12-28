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
MOCK_SEED = 100


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

def plot_results(results_left, results_right):
    import matplotlib.font_manager as fm
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

    # Register font
    font_path = f"{current_dir}/LinLibertine_R.ttf"
    if font_path not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path)

    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    # Two panels only:
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

def best_feasible_perf_until(results_dict, solution_key, until_time=TUNING_BUDGET):
    """
    Return (best_perf, best_time) where best_perf is the best objective metric
    among trials that satisfy the constraint and have tuning_time <= until_time.
    If nothing feasible, returns (None, None).
    """
    sol_list = results_dict["results"].get(solution_key, [])
    if not sol_list:
        return None, None

    best_perf = None
    best_t = None

    for hp, perf in sol_list:
        t = perf[0]
        if t > until_time:
            continue

        recall = perf[1]
        qps = perf[2]

        # constraint check + objective metric selection
        if results_dict["recall_min"] is not None:
            if recall < results_dict["recall_min"]:
                continue
            metric = qps
        else:
            if qps < results_dict["qps_min"]:
                continue
            metric = recall

        if best_perf is None or metric > best_perf:
            best_perf = metric
            best_t = t

    return best_perf, best_t

def first_time_reach_threshold(results_dict, solution_key, threshold):
    """
    Return earliest tuning_time t such that (constraint satisfied) and (metric >= threshold).
    If never reached, return None.
    """
    sol_list = results_dict["results"].get(solution_key, [])
    if not sol_list:
        return None

    sol_list = sorted(sol_list, key=lambda r: r[1][0])  # chronological

    for hp, perf in sol_list:
        t = perf[0]
        recall = perf[1]
        qps = perf[2]

        if results_dict["recall_min"] is not None:
            if recall < results_dict["recall_min"]:
                continue
            metric = qps
        else:
            if qps < results_dict["qps_min"]:
                continue
            metric = recall

        if metric >= threshold:
            return t

    return None

def analyze_panel(results_dict, panel_name):
    oracle_best, oracle_best_t = best_feasible_perf_until(
        results_dict, "brute_force", until_time=TUNING_BUDGET
    )
    if oracle_best is None:
        print(f"[{panel_name}] Oracle has no feasible point (check constraints or data).")
        return

    target95 = 0.95 * oracle_best

    print(f"\n=== {panel_name} ===")
    print(f"Oracle best (feasible, <=budget): {oracle_best:.6g} @ t={oracle_best_t:.3f}s")
    print(f"95% target: {target95:.6g}")

    for sol in results_dict["results"].keys():
        best, best_t = best_feasible_perf_until(results_dict, sol, until_time=TUNING_BUDGET)
        t95 = first_time_reach_threshold(results_dict, sol, target95)

        if best is None:
            best_str = "None"
            ratio_str = "None"
        else:
            ratio = (best / oracle_best) * 100.0
            best_str = f"{best:.6g} @ t={best_t:.3f}s"
            ratio_str = f"{ratio:.2f}%"

        t95_str = "None" if t95 is None else f"{t95:.3f}s"

        print(
            f"- {sol:12s} | best@budget: {best_str:>22s} | oracle%: {ratio_str:>8s} | first95%: {t95_str}"
        )


def analyze_results(left_results, right_results):
    analyze_panel(left_results,  "Recall_min mode (maximize QPS)")
    analyze_panel(right_results, "QPS_min mode (maximize Recall)")


def main():
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

    IMPL = "milvus"
    DATASET = "nytimes-256-angular"
    # DATASET = "glove-100-angular"
    DATASET = "sift-128-euclidean"
    SAMPLING_COUNT = 10

    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"  # choose "q50", "q75", "q90", ...
    QPS_MIN = get_qps_metrics_dataset(IMPL, DATASET, ret_dict=True)[QPS_MIN_KEY]
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
    plot_results(results_left, results_right)
    analyze_results(results_left, results_right)


if __name__ == "__main__":
    main()
