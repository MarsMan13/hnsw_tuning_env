from concurrent.futures import ProcessPoolExecutor, as_completed
from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN, M_MIN, M_MAX, EFC_MIN, EFC_MAX
from src.solutions import print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = GroundTruth(impl, dataset)
    results = []
    for M in range(M_MIN, M_MAX + 1, 1):
        for efC in range(EFC_MIN, EFC_MAX + 1, 4):
            efS = gd.get_efS(M, efC, recall_min, qps_min, method="binary")
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            results.append(((M, efC, efS), (0.0, recall, qps, total_time, build_time, index_size)))
    return results

def process_dataset(dataset, impl, tuning_budget):
    RECALL_MIN = 0.95
    QPS_MIN = get_qps_metrics_dataset(impl, dataset, ret_dict=True)["q90"]

    output_line = f"{dataset},"

    results = run(impl, dataset, RECALL_MIN, None, tuning_budget)
    opt, _ = print_optimal_hyperparameters(results, recall_min=RECALL_MIN, qps_min=None)
    output_line += f"{str(opt[0]).replace(' ', '')},{opt[1][2]},"

    results = run(impl, dataset, None, QPS_MIN, tuning_budget)
    opt, _ = print_optimal_hyperparameters(results, recall_min=None, qps_min=QPS_MIN)
    output_line += f"{str(opt[0]).replace(' ', '')},{opt[1][1]}"

    return output_line

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

    output_lines = []
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(process_dataset, dataset, IMPL, TUNING_BUDGET)
            for dataset in DATASETS
        ]

        for future in as_completed(futures):
            output_lines.append(future.result())

    output = "\n".join(output_lines)
    print(output.strip())

if __name__ == "__main__":
    main()
