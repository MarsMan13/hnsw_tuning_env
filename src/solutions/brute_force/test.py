from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl, dataset)
    results = []
    for M in range(48, 48+1, 2):
    # for M in [32, 40]:
        for efC in range(16, 1024+1, 2):
            efS = gd.get_efS(M, efC, recall_min, method="binary")
            # efS = gd.get_efS(M, efC, recall_min, method="linear")
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            gd.tuning_time += 100.0
            if qps != 0.0:
                print(f"({M}, {efC}, {efS})-{round(qps.item())}")
            results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
    return results
    
if __name__ == "__main__":
    tuning_budget = float("inf")
    for RECALL_MIN in [0.95]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, tuning_budget)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                # postprocess_results(results, solution="brute_force", impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=tuning_budget)
