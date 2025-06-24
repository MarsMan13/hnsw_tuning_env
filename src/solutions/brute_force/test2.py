from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl, dataset)
    results = []
    # for M in range(32, 32+1, 1):
    for M in [8]:
        for efS in [512]:
            for efC in range(64, 1024+1, 8):
                # efS = gd.get_efS(M, efC, recall_min, method="binary")
                # efS = gd.get_efS(M, efC, recall_min, method="linear")
                recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
                gd.tuning_time += 100.0
                # if qps != 0.0:
                #     print(f"({M}, {efC}, {efS})-{round(qps.item())}")
                results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
    for result in results:
        print(f"({result[0][0]}, {result[0][1]}, {result[0][2]})|{round(result[1][1], 2)}|{round(result[1][2])}")
    return results
    
if __name__ == "__main__":
    tuning_budget = float("inf")
    for RECALL_MIN in [0.95]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, tuning_budget)
                global_opt_hp, local_opt_hps = print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                for hp, perf in local_opt_hps:
                    if hp:
                        print(f"x={hp[0]}, y={hp[1]}, z={hp[2]}")
                # postprocess_results(results, solution="brute_force", impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=tuning_budget)
