from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from src.constants import IMPL, DATASET, SEED, TUNING_BUDGET, RECALL_MIN
import random

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    if not recall_min and not qps_min:
        raise ValueError("Either recall_min or qps_min must be specified.")
    if recall_min and qps_min:
        raise ValueError("Only one of recall_min or qps_min should be specified.")
    gd = GroundTruth(impl=impl, dataset=dataset)
    random.seed(SEED)
    results = []
    while True:
        M = random.randint(4, 64)
        efC = random.randint(8, 512)
        efS = random.randint(10, 1024)
        recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS)
        if gd.tuning_time > tuning_budget:
            print(f"Tuning time out")
            break
        results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
    return results

def recall_min():   
    for RECALL_MIN in [0.90, 0.95, 0.975]:
        for IMPL in ["hnswlib", "faiss"]:
        # for IMPL in ["hnswlib"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(results, solution="random_search", impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)

def qps_min():
    for QPS_MIN in [2500, 5000, 10000, 25000]:
        for IMPL in ["hnswlib", "faiss"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
                results = run(IMPL, DATASET, qps_min=QPS_MIN, tuning_budget=TUNING_BUDGET)
                print_optimal_hyperparameters(results, qps_min=QPS_MIN)
                postprocess_results(results, solution="random_search", impl=IMPL, dataset=DATASET, qps_min=QPS_MIN, tuning_budget=TUNING_BUDGET)

if __name__ == "__main__":
    recall_min()
    qps_min()
