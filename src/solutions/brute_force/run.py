from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN, M_MIN, M_MAX, EFC_MIN, EFC_MAX, EFS_MIN, EFS_MAX, SEED
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from joblib import Memory

# memory = Memory("/tmp/brute_force_cache", verbose=0)
# @memory.cache
def run(impl=IMPL, dataset=DATASET, recall_min=None , qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = GroundTruth(impl, dataset)
    results = []
    for M in range(M_MIN, M_MAX+1, 2):
        for efC in range(EFC_MIN, EFC_MAX//2+1, 2):
            # efS = gd.get_efS(M, efC, target_recall=recall_min, target_qps=qps_min, method="linear")
            efS = gd.get_efS(M, efC, target_recall=recall_min, target_qps=qps_min, method="binary")
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            gd.tuning_time += 100.0
            results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
        # print_optimal_hyperparameters(results, recall_min=recall_min, qps_min=qps_min)
    return results

def recall_min():
    tuning_budget = float("inf")
    # for RECALL_MIN in [0.90, 0.95, 0.975]:
    for RECALL_MIN in [0.95]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["sift-128-euclidean"]:
            for DATASET in ["nytimes-256-angular"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular"]:
                results = run(IMPL, DATASET, recall_min=RECALL_MIN, tuning_budget=tuning_budget)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                # postprocess_results(results, solution="brute_force", impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=tuning_budget)

def qps_min():
    tuning_budget = float("inf")
    for QPS_MIN in [2500, 5000, 10000, 25000]:
    # for QPS_MIN in [2500]:
        for IMPL in ["hnswlib", "faiss"]:
        # for IMPL in ["faiss"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, qps_min=QPS_MIN, tuning_budget=tuning_budget)
                print_optimal_hyperparameters(results, qps_min=QPS_MIN)
                postprocess_results(results, solution="brute_force", impl=IMPL, dataset=DATASET, qps_min=QPS_MIN, tuning_budget=tuning_budget)


if __name__ == "__main__":
    recall_min()
    # qps_min()