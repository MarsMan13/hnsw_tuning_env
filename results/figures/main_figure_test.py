import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    plot_gradient_bar_with_oracle_on_ax,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
MOCK_SEED = 0


def short_ds(name: str) -> str:
    return name.split("-")[0]


def get_results(impl, dataset, solutions, recall_min=None, qps_min=None, sampling_count=None, tuning_time=TUNING_BUDGET):
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

    return {"impl": impl, "dataset": dataset, "recall_min": recall_min, "qps_min": qps_min, "results": results_combi}


def main():
    # Font
    font_path = f"{current_dir}/LinLibertine_R.ttf"
    if font_path not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "optuna", "nsga"]
    SOLUTIONS = ["grid_search"]
    IMPLS = ["hnswlib", "faiss"]
    IMPLS = ["hnswlib"]
    DATASETS = ["nytimes-256-angular", "glove-100-angular", "sift-128-euclidean", "youtube-1024-angular", "deep1M-256-angular"]
    DATASETS = ["glove-100-angular"]
    SAMPLING_COUNT = [10]
    RECALL_MINS = [0.95]
    QPS_MIN_KEY = "q75"

    tasks = []
    for impl in IMPLS:
        for metric in ["recall_min"]:
            for dataset in DATASETS:
                for sampling_count in SAMPLING_COUNT:
                    if metric == "recall_min":
                        for recall_min in RECALL_MINS:
                            tasks.append((impl, dataset, SOLUTIONS, recall_min, None, sampling_count))
                    else:
                        qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
                        tasks.append((impl, dataset, SOLUTIONS, None, qps_min, sampling_count))

    results = [get_results(*t) for t in tasks]
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
