from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl, dataset)
    results = []
    # for M in range(32, 32+1, 1):
    for M in range(8, 64+1, 4):    
        opt_efC = 0.0
        opt_efS = 0.0 
        opt_qps = 0.0
        for efC in range(64, 1024+1, 8):
            efS = gd.get_efS(M, efC, recall_min, method="binary")
            # efS = gd.get_efS(M, efC, recall_min, method="linear")
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            if not recall:  recall, qps = 0, 0
            gd.tuning_time += 100.0
            if qps > opt_qps:
                opt_efC = efC
                opt_efS = efS
                opt_qps = qps
            opt_qps = round(float(opt_qps.item()) if not isinstance(opt_qps, float) else opt_qps, 0)
        results.append((M, opt_efC, opt_efS, opt_qps))
    return results
    
if __name__ == "__main__":
    tuning_budget = float("inf")
    for RECALL_MIN in [0.95]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "youtube-1024-angular"]:
                print(DATASET)
                print("index M efC efS perf rank\n", end="")
                results = run(IMPL, DATASET, RECALL_MIN, tuning_budget)
                sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
                index = 1
                for result in results:
                    print(f"{index} {result[0]} {result[1]} {result[2]} {result[3]} {sorted_results.index(result) + 1}")
                    index+=1