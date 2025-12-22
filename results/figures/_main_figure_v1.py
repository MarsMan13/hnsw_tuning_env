import matplotlib.pyplot as plt

from src.constants import TUNING_BUDGET
from src.utils import filename_builder, get_optimal_hyperparameter, \
    load_search_results, plot_accumulated_timestamp_on_ax
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

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
    #* 1. Load results for all solutions under the given condition
    for solution in solutions:
        filename = filename_builder(
            solution, impl, dataset, recall_min, qps_min
        )
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)
        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(
                results, recall_min=recall_min, qps_min=qps_min
            )
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            optimal_hp = (hp, perf)
            results = [optimal_hp]  # For brute_force, we only keep the optimal hyperparameter
        else:
            results = [result for result in results if result[1][0] <= tuning_time]  # Filter results by tuning time
        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results":results_combi
    }

def main():
    SOLUTIONS = [
        "brute_force",
        "our_solution",
        "grid_search",
        "random_search",
        "vd_tuner",
        "optuna",
        "nsga",
    ]
    IMPLS = [
        "faiss",
        "hnswlib",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "youtube-1024-angular",
        "deep1M-256-angular",
    ]
    SAMPLING_COUNT = [
        10,
    ]
    RECALL_MINS = [0.975]
    # --- Start of multiprocessing modification ---

    #* 1. Create a list to hold all the tasks to be executed.
    # A task is a tuple of arguments for the _process_single_metric function.
    tasks = []
    # 1st: IMPLS 순서 = [hnswlib, faiss]
    for impl in IMPLS:
        # 2nd: metric 종류 (recall_min, qps_min)
        for metric in ["recall_min", "qps_min"]:
            for dataset in DATASETS:
                for sampling_count in SAMPLING_COUNT:
                    if metric == "recall_min":
                        for recall_min in RECALL_MINS:
                            task_args = (impl, dataset, SOLUTIONS, recall_min, None, sampling_count)
                            tasks.append(task_args)
                    else:  # qps_min
                        qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)["q90"]
                        task_args = (impl, dataset, SOLUTIONS, None, qps_min, sampling_count)
                        tasks.append(task_args)
    results = []
    for impl, dataset, solutions, recall_min, qps_min, sampling_count in tasks:
        results.append(get_results(
            impl=impl,
            dataset=dataset,
            solutions=solutions,
            recall_min=recall_min,
            qps_min=qps_min,
            sampling_count=sampling_count
        ))
    print(len(results))
    ## TODO
    # 4x5 subplot 예시
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))  #! <- Original
    # fig, axes = plt.subplots(4, 5, figsize=(22, 12))

    # (실제 결과 반복문)
    for ax, result in zip(axes.flat, results):
        plot_accumulated_timestamp_on_ax(
            ax, result["results"], result["recall_min"], result["qps_min"]
        )
        # metric = "recall" if result["recall_min"] is not None else "qps"
        # ax.set_title(f"{result['impl']} | {result['dataset']} | {metric}")

    # 1️⃣ 모든 subplot의 핸들/라벨 모으기
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # 2️⃣ 중복 제거
    by_label = dict(zip(labels, handles))

    # 3️⃣ 범례를 Figure 위쪽에 띄우되, Figure 안쪽이 아니라 조금 바깥에 두기
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),   # Figure 영역을 벗어나 위쪽 10% 공간에 배치
        ncol=5,
        fontsize=18,
        frameon=False
    )

    # 4️⃣ subplots 간격 수동 조정
    plt.subplots_adjust(
        top=0.95,   # 위쪽 subplot 영역을 90%까지만 사용하여 legend 영역 확보
        wspace=0.3,
        hspace=0.3
    )

    fig.savefig("main_figure.pdf", bbox_inches="tight")
    plt.show() 

if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly,
    # especially on Windows and macOS. It prevents child processes from
    # re-importing and re-executing the main script's code.
    main()