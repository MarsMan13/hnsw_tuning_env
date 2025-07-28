from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf: perf[2] if recall_min else perf[1]
    gd = GroundTruth(impl, dataset)
    results = []
    # M, efC = 32, 256
    M, efC = 64, 1024 
    print("M,efC,efS,recall,qps")
    for efS in range(1024, 32, -96):
        recall, qps, *_ = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
        print(f"{M},{efC},{efS},{round(recall,5)},{round(qps)}")
    return results

def main():
    tuning_budget = float("inf")
    for RECALL_MIN in [0.95]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular"]:
            for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, None, tuning_budget)
                sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
                index = 1
                for result in results:
                    print(f"{index} {result[0]} {result[1]} {result[2]} {result[3]} {sorted_results.index(result) + 1}")
                    index+=1

if __name__ == "__main__":
    main()