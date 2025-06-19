from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from src.constants import EFC_MAX, EFC_MIN, EFS_MAX, EFS_MIN, M_MAX, M_MIN 
from src.constants import IMPL, DATASET, SEED, TUNING_BUDGET, RECALL_MIN
import random
from tqdm import tqdm

STEP_M = 2
STEP_EFC = 16
STEP_EFS = 16

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    if not recall_min and not qps_min:
        raise ValueError("Either recall_min or qps_min must be specified.")
    if recall_min and qps_min:
        raise ValueError("Only one of recall_min or qps_min should be specified.")
    gd = GroundTruth(impl=impl, dataset=dataset)
    random.seed(SEED)
    results = []
    candidates = [
        (M, efC)
        for M in range(M_MIN, M_MAX + 1, STEP_M)
        for efC in range(EFC_MIN, EFC_MAX + 1, STEP_EFC)
        if M <= efC
    ]
    random.shuffle(candidates)  # Shuffle candidates to ensure randomness in the search order
    for M, efC in tqdm(candidates, desc=f"GridSearch[{impl}|{dataset}]", unit="config"):
        if recall_min:
            efS = gd.get_efS(M, efC, target_recall=recall_min)
        elif qps_min:
            efS = gd.get_efS(M, efC, target_qps=qps_min)
        if gd.tuning_time > tuning_budget:
            print(f"Tuning time out at {gd.tuning_time:.2f}s")
            break
        perf = gd.get(M=M, efC=efC, efS=efS)
        results.append(((M, efC, efS), (gd.tuning_time, *perf)))
    return results

def recall_min():    
    for RECALL_MIN in [0.90, 0.95, 0.975]:
        for IMPL in ["hnswlib", "faiss"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results,
                    solution="grid_search_heuristic",
                    impl=IMPL,
                    dataset=DATASET,
                    recall_min=RECALL_MIN,
                    tuning_budget=TUNING_BUDGET
                )

def qps_min():
    for QPS_MIN in [2500, 5000, 10000, 25000]:
        for IMPL in ["hnswlib", "faiss"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
                results = run(IMPL, DATASET, qps_min=QPS_MIN, tuning_budget=TUNING_BUDGET)
                print_optimal_hyperparameters(results, qps_min=QPS_MIN)
                postprocess_results(
                    results,
                    solution="grid_search_heuristic",
                    impl=IMPL,
                    dataset=DATASET,
                    qps_min=QPS_MIN,
                    tuning_budget=TUNING_BUDGET
                )

if __name__ == "__main__":
    recall_min()
    qps_min()