import os
import matplotlib.pyplot as plt

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    plot_accumulated_timestamp_on_ax,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"

MOCK_SEED = "0_cherry"

def get_results(
    impl: str,
    dataset: str,
    solutions: list,
    recall_min: float = None,
    qps_min: int = None,
    sampling_count: int = None,
    tuning_time: int = TUNING_BUDGET,
):
    assert (recall_min is not None) != (qps_min is not None), "Either recall_min or qps_min must be specified, but not both."
    results_combi = {}
    max_perf = None
    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)
        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            results = [(hp, perf)]
            max_perf = perf
        else:
            results = [r for r in results if r[1][0] <= tuning_time]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "max_perf": max_perf[2] if recall_min else max_perf[1]
    }


# =========================
# Analysis helpers (same logic as your milvus script)
# =========================

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

        if results_dict["recall_min"] is not None:
            # feasible requires recall >= recall_min
            if recall < results_dict["recall_min"]:
                continue
            metric = qps  # objective
        else:
            # feasible requires qps >= qps_min
            if qps < results_dict["qps_min"]:
                continue
            metric = recall  # objective

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


def analyze_panel(results_dict, panel_name, print_only=None):
    """
    print_only: Optional set/list of solution keys to print. If None, print all.
    """
    oracle_best, oracle_best_t = best_feasible_perf_until(
        results_dict, "brute_force", until_time=TUNING_BUDGET
    )
    if oracle_best is None:
        print(f"[{panel_name}] Oracle has no feasible point.")
        return

    target95 = 0.95 * oracle_best

    print(f"\n=== {panel_name} ===")
    print(f"Oracle best (feasible, <=budget): {oracle_best:.6g} @ t={oracle_best_t:.3f}s")
    print(f"95% target: {target95:.6g}")

    keys = list(results_dict["results"].keys())
    if print_only is not None:
        keys = [k for k in keys if k in set(print_only)]

    for sol in keys:
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


def analyze_all_results(results_list):
    """
    results_list: list of results_dict produced by get_results
    """
    for r in results_list:
        impl = r["impl"]
        dataset = r["dataset"]

        if r["recall_min"] is not None:
            panel = f"{impl} / {dataset} / recall_min={r['recall_min']} (maximize QPS)"
        else:
            panel = f"{impl} / {dataset} / qps_min={r['qps_min']} (maximize Recall)"

        analyze_panel(r, panel_name=panel)


# =========================
# Main (plot + analysis)
# =========================

def main():
    import matplotlib.font_manager as fm

    font_path = f"{current_dir}/LinLibertine_R.ttf"
    if font_path not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path)

    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    SOLUTIONS = [
        "brute_force", "our_solution", "grid_search",
        "random_search", "vd_tuner", "optuna", "nsga",
    ]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = [
        "nytimes-256-angular", "glove-100-angular", "sift-128-euclidean",
        "youtube-1024-angular", "deep1M-256-angular",
    ]
    COLUMN_LABELS = ["nytimes", "glove", "sift", "deep1M", "youtube"]

    SAMPLING_COUNT = [10]
    RECALL_MINS = [0.95]
    QPS_MIN_KEY = "q75"

    tasks = []
    for impl in IMPLS:
        for metric in ["recall_min", "qps_min"]:
            for dataset in DATASETS:
                for sampling_count in SAMPLING_COUNT:
                    if metric == "recall_min":
                        for recall_min in RECALL_MINS:
                            tasks.append((impl, dataset, SOLUTIONS, recall_min, None, sampling_count))
                    else:
                        qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
                        tasks.append((impl, dataset, SOLUTIONS, None, qps_min, sampling_count))

    results = []
    for impl, dataset, solutions, recall_min, qps_min, sampling_count in tasks:
        results.append(
            get_results(
                impl=impl,
                dataset=dataset,
                solutions=solutions,
                recall_min=recall_min,
                qps_min=qps_min,
                sampling_count=sampling_count,
            )
        )

    # ===== Plot (unchanged) =====
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))

    for ax, result in zip(axes.flat, results):
        plot_accumulated_timestamp_on_ax(ax, result["results"], result["recall_min"], result["qps_min"], max_perf=result["max_perf"])

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=6,
        fontsize=18,
        frameon=False,
    )

    fig.text(0.02, 0.70, "hnswlib", va="center", ha="center", rotation="vertical", fontsize=24)
    fig.text(0.02, 0.30, "faiss", va="center", ha="center", rotation="vertical", fontsize=24)

    for i, label in enumerate(COLUMN_LABELS):
        axes[3, i].set_xlabel(label, fontsize=20, labelpad=10)

    plt.subplots_adjust(left=0.06, bottom=0.1, top=0.94, wspace=0.3, hspace=0.4)

    fig.savefig("main_figure.pdf", bbox_inches="tight")
    plt.show()

    # ===== Analysis (new) =====
    analyze_all_results(results)


if __name__ == "__main__":
    main()
