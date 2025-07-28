from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf: perf[2] if recall_min else perf[1]
    gd = GroundTruth(impl, dataset)
    results = []
    M = 32 if impl == "faiss" else 16
    efC = 40 if impl == "faiss" else 200
    efS = gd.get_efS(M, efC, recall_min, qps_min, method="binary")
    if efS == 0:
        efS = 1024 if recall_min else 10
    recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
    results.append(((M, efC, efS), (0.0, recall, qps, total_time, build_time, index_size)))
    return results

def main():
    # IMPL = "hnswlib"
    IMPL = "faiss"
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "deep1M-256-angular",
        "youtube-1024-angular",
    ]
    output = ""
    for DATASET in DATASETS:
        RECALL_MIN = 0.95
        QPS_MIN = get_qps_metrics_dataset(IMPL, DATASET, ret_dict=True)["q90"]
        output += f"{DATASET},"
        results = run(IMPL, DATASET, RECALL_MIN, None, TUNING_BUDGET)
        opt, _ = print_optimal_hyperparameters(results, recall_min=0.0, qps_min=None)
        output += f"{str(opt[0]).replace(' ', '')},{opt[1][2]},"
        results = run(IMPL, DATASET, None, QPS_MIN, TUNING_BUDGET)
        opt, _ = print_optimal_hyperparameters(results, recall_min=None, qps_min=0.0)
        output += f"{str(opt[0]).replace(' ', '')},{opt[1][1]}\n\n"
    print(output.strip())

if __name__ == "__main__":
    main()