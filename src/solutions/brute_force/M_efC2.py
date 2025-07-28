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
    M = 64 
    print(f"{recall_min} {qps_min}")
    print("M,efC,efS,recall,qps")
    for efC in [64, 128, 256, 512, 1024]:
        efS = gd.get_efS(M, efC, recall_min, qps_min, method="binary")
        recall, qps, *_, index_size = gd.get(M, efC, efS, tracking_time=False)
        recall = recall if not hasattr(recall, "item") else recall.item()
        qps = qps if not hasattr(qps, "item") else qps.item()
        print(f"{M},{efC},{efS},{round(recall,4)},{round(qps)}")
    return results

def main():
    IMPL = "faiss"
    RECALL_MIN = 0.95
    QPS_MIN = 10000
    # DATASET = "nytimes-256-angular"
    DATASET = "glove-100-angular"
    _ = run(IMPL, DATASET, RECALL_MIN, None, TUNING_BUDGET)
    _ = run(IMPL, DATASET, None, QPS_MIN, TUNING_BUDGET)

if __name__ == "__main__":
    main()