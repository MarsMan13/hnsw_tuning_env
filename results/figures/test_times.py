from src.utils import load_search_results
from src.solutions import print_optimal_hyperparameters
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset


def worker(params, perf=None):
    impl, dataset, solution, recall_min, qps_min, time = params
    get_perf = lambda x: int(x[1][2]) if recall_min else round(x[1][1],4)  # Recall or QPS
    try:
        if recall_min:
            filename = f"{solution}_{impl}_{dataset}_{recall_min}r_{None}q.csv"
        else:
            filename = f"{solution}_{impl}_{dataset}_{None}r_{qps_min}q.csv"
        results = load_search_results(solution, filename)
        if solution != "brute_force":
            results = [result for result in results if result[1][0] <= time * 3600]  # Filter by tuning time
        if not results:
            return time * 3600, 0.0
    except Exception as e:
        return f"Error loading {filename}: {e}"
    if perf is None:
        opt_hp, _ = print_optimal_hyperparameters(results, recall_min=recall_min, qps_min=qps_min)
    else:
        filtered_results = [result for result in results if get_perf(result) >= perf]
        sorted_results = sorted(filtered_results, key=lambda x: x[1][0])
        if not sorted_results:
            return time * 3600, 0.0
        return int(sorted_results[0][1][0]), get_perf(sorted_results[0])
    return int(opt_hp[1][0]), get_perf(opt_hp) 

def main():
    IMPLS = [
        "hnswlib",
        "faiss",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "deep1M-256-angular",
        "youtube-1024-angular",
    ]
    SOLUTIONS = [
        "brute_force",
        # # "our_solution",
        # "random_search",
        # "grid_search",
        # "vd_tuner",
    ]
    recall_min = 0.95
    qps_min = "q75"
    tuning_time = 4
    stdout = ""
    for impl in IMPLS:
        line = f"{impl},"
        for dataset in DATASETS:
            max_time, max_perf = tuning_time * 3600, 0.0
            for solution in SOLUTIONS:
                time, perf = worker((impl, dataset, solution, recall_min, None, tuning_time))
                if max_perf < perf:
                    max_time, max_perf = time, perf
            solution_time, _ = worker((impl, dataset, "our_solution", recall_min, None, tuning_time), perf=max_perf)
            line += f"{solution_time/max_time},"
        stdout += line + "\n"
    stdout += "\n"
    for impl in IMPLS:
        line = f"{impl},"
        for dataset in DATASETS:
            max_time, max_perf = tuning_time * 3600, 0.0
            qps = get_qps_metrics_dataset(impl=impl, dataset=dataset, ret_dict=True)[qps_min]
            for solution in SOLUTIONS:
                time, perf = worker((impl, dataset, solution, None, qps, tuning_time))
            solution_time, _ = worker((impl, dataset, "our_solution", None, qps, tuning_time), perf=max_perf)
            line += f"{solution_time/max_time},"
        stdout += line + "\n"
    print(stdout)

if __name__ == "__main__":
    main()