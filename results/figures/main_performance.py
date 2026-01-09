from src.utils import load_search_results
from src.solutions import print_optimal_hyperparameters
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

MOCK_SEED = "0_cherry"

def worker(params):
    impl, dataset, solution, recall_min, qps_min, time = params
    get_perf = lambda x: int(x[1][2]) if recall_min else round(x[1][1],4)  # Recall or QPS
    try:
        if recall_min:
            filename = f"{solution}_{impl}_{dataset}_{recall_min}r_{None}q.csv"
        else:
            filename = f"{solution}_{impl}_{dataset}_{None}r_{qps_min}q.csv"
        results = load_search_results(solution, filename, seed=MOCK_SEED)
        if solution != "brute_force":
            results = [result for result in results if result[1][0] <= time * 3600]  # Filter by tuning time
        if not results:
            return time * 3600, 0.0
    except Exception as e:
        return f"Error loading {filename}: {e}"
    opt_hp, _ = print_optimal_hyperparameters(results, recall_min=recall_min, qps_min=qps_min)
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
        "our_solution",
        "random_search",
        "grid_search",
        "vd_tuner",
        "optuna",
        "nsga",
    ]
    RECALL_MINS = [0.95]
    QPS_MINS = ["q75"]
    tuning_time = 4
    stdout = ""
    for impl in IMPLS:
        for solution in SOLUTIONS:
            line = f"{solution},"
            for dataset in DATASETS:
                for recall_min in RECALL_MINS:
                    time, perf = worker((impl, dataset, solution, recall_min, None, tuning_time))
                    line += f"{time},{perf},"
            line += "\n"
            stdout += line
    stdout += "\n"
    for impl in IMPLS:
        for solution in SOLUTIONS:
            line = f"{solution},"
            for dataset in DATASETS:
                for qps_min in QPS_MINS:
                    qps = get_qps_metrics_dataset(impl=impl, dataset=dataset, ret_dict=True)[qps_min]
                    time, perf = worker((impl, dataset, solution, None, qps, tuning_time))
                    line += f"{time},{perf},"
            line += "\n"
            stdout += line
    print(stdout)

if __name__ == "__main__":
    main()