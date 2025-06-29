from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf: perf[2] if recall_min else perf[1]
    gd = GroundTruth(impl, dataset)
    results = []
    # for M in range(32, 32+1, 1):
    for M in range(8, 64+1, 8):    
        # opt_efC = 0.0
        # opt_efS = 0.0 
        # opt_qps = 0.0
        # for efC in range(64, 1024+1, 8):
            # efS = gd.get_efS(M, efC, recall_min, qps_min, method="binary")
            # efS = gd.get_efS(M, efC, recall_min, method="linear")
            # recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            # if not recall:  recall, qps = 0, 0
            # gd.tuning_time += 100.0
            # if qps > opt_qps:
            #     opt_efC = efC
            #     opt_efS = efS
            #     opt_qps = qps
        #     opt_qps = round(float(opt_qps.item()) if not isinstance(opt_qps, float) else opt_qps, 0)
        # results.append((M, opt_efC, opt_efS, opt_qps))
        for efC in range(64, 1024+1, 8):
            efS = gd.get_efS(M, efC, recall_min, qps_min, method="binary")
            perf = get_perf(gd.get(M=M, efC=efC, efS=efS, tracking_time=False))
            results.append(((M, efC, efS), perf))
    return results

def run_recall_min():
    tuning_budget = float("inf")
    for RECALL_MIN in [0.95]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "youtube-1024-angular"]:
                print(DATASET)
                print("index M efC efS perf rank\n", end="")
                results = run(IMPL, DATASET, RECALL_MIN, None, tuning_budget)
                sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
                index = 1
                for result in results:
                    print(f"{index} {result[0]} {result[1]} {result[2]} {result[3]} {sorted_results.index(result) + 1}")
                    index+=1

def run_qps_min():
    tuning_budget = float("inf")
    for IMPL in ["faiss"]:
        # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "youtube-1024-angular"]:
        for DATASET in ["nytimes-256-angular"]:
            for QPS_MIN in get_qps_metrics_dataset(impl=IMPL, dataset=DATASET)[:1]:
                print("index M efC efS perf rank\n", end="")
                results = run(IMPL, DATASET, None, QPS_MIN, tuning_budget)
                # sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
                index = 1
                for result in results:
                    # print(f"{index} {result[0]} {result[1]} {result[2]} {result[3]} {sorted_results.index(result) + 1}")
                    print(f"{result[0][0]} {result[0][1]} {result[0][2]} {result[1]}")
                    index+=1

if __name__ == "__main__":
    run_qps_min()