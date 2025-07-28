from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN, M_MIN, M_MAX, EFC_MIN, EFC_MAX, EFS_MIN, EFS_MAX, SEED
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def run(impl=IMPL, dataset=DATASET, recall_min=None , qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = GroundTruth(impl, dataset)
    opt_results = []
    for M in range(8, M_MAX+1, 2):
        results = []
        for efC in range(EFC_MIN, EFC_MAX+1, 16):
            # efS = gd.get_efS(M, efC, target_recall=recall_min, target_qps=qps_min, method="linear")
            efS = gd.get_efS(M, efC, target_recall=recall_min, target_qps=qps_min, method="binary")
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
        opt_hp, _ = print_optimal_hyperparameters(results, recall_min=recall_min, qps_min=qps_min)
        if opt_hp[0][0] is None:
            opt_results.append((M, 1024, 1024, 0, 0))
            continue 
        M, efC, efS = opt_hp[0]
        _, recall, qps, *_ = opt_hp[1]
        opt_results.append((M, efC, efS, recall, qps))
    print("M,efC,efS,recall,qps")
    for M, efC, efS, recall, qps in opt_results:
        print(f"{M},{efC},{efS},{round(recall, 4)},{round(qps)}")
    return results

def main():
    IMPL = "faiss"
    RECALL_MIN = 0.95
    QPS_MIN = 10000
    DATASET = "nytimes-256-angular"
    # DATASET = "glove-100-angular"
    _ = run(IMPL, DATASET, RECALL_MIN, None, TUNING_BUDGET)
    _ = run(IMPL, DATASET, None, QPS_MIN, TUNING_BUDGET)

if __name__ == "__main__":
    main()