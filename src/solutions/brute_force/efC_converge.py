from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf: perf[2] if recall_min else perf[1]
    gd = GroundTruth(impl, dataset)
    results = []
    efS = 512
    print("efC,Recall")
    for M in [8, 56]:
        for efC in range(64+96, 1024+1, 96):
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            # print(f"{M},{efC},{efS},{int(index_size)}")
            print(f"{efC},{round(recall, 4)}")
    return results

def main()
    IMPL = "hnswlib"
    RECALL_MIN = 0.95
    QPS_MIN = 10000
    DATASET = "nytimes-256-angular"
    _ = run(IMPL, DATASET, RECALL_MIN, None, TUNING_BUDGET)
    # _ = run(IMPL, DATASET, None, QPS_MIN, TUNING_BUDGET)

if __name__ == "__main__":
    main()