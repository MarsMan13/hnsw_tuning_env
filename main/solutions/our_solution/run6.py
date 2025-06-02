import math
import random

from numpy import append

from main.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, SEED, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX, M_MIN, M_MAX
from main.solutions import postprocess_results, print_optimal_hyperparameters
from main.solutions.our_solution import get_max_qps_of_M, get_exploitation_targets
from static.ground_truths.ground_truth import GroundTruth

def exploration_phase(results, ground_truth, recall_min, exploration_budget):
    gd = ground_truth 
    efC_left, efC_right = EFC_MIN, EFC_MAX 
    efS_min, efS_max = EFS_MIN, EFS_MAX
    M_left, M_right = M_MIN, M_MAX
    appended_hp = set()
    try:
        while M_right - M_left > 3:
            M_mid1 = M_left + (M_right - M_left) // 3
            M_mid2 = M_right - (M_right - M_left) // 3
            for M in [M_mid1, M_mid2]:
                efC_left = max(EFC_MIN, int(efC_left * 0.75))
                efC_right = min(EFC_MAX, int(efC_right * 1.25))
                print(f"{M:2} | {efC_left:3} {efC_right:3}")
                efS_local_min = efS_min
                # efS_local_max = min(efS_max, EFS_MAX) #! TODO
                efS_local_max = EFS_MAX 
                efC_iter_limit = math.ceil(math.log(EFC_MAX - EFC_MIN, 2.5)) // 3   #! TODO
                efC_count = 0
                while efC_right - efC_left > 3 and efC_count < efC_iter_limit:
                    efC_count += 1
                    efC_mid1 = efC_left + (efC_right - efC_left) // 3
                    efC_mid2 = efC_right - (efC_right - efC_left) // 3
                    print(f"   {M:2} | {efC_left:3} {efC_mid1:3} {efC_mid2:3} {efC_right:3}")
                    efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=efS_local_min, efS_max=efS_local_max)
                    efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=efS_local_min, efS_max=efS_mid2, skip_time=True)
                    if gd.tuning_time > exploration_budget:  
                        gd.rollback() 
                        raise TimeoutError("tuning time out")
                    perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
                    perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
                    if (M, efC_mid1, efS_mid1) not in appended_hp:
                        appended_hp.add((M, efC_mid1, efS_mid1))
                        results.append(((M, efC_mid1, efS_mid1), (gd.tuning_time, *perf_mid1)))
                    if (M, efC_mid2, efS_mid2) not in appended_hp:
                        results.append(((M, efC_mid2, efS_mid2), (gd.tuning_time, *perf_mid2)))
                        appended_hp.add((M, efC_mid2, efS_mid2))
                    if perf_mid1[1] <= perf_mid2[1]:
                        efC_left = efC_mid1
                        efS_local_min = efS_mid1
                    else:
                        efC_right = efC_mid2
                        efS_local_max = efS_mid2
            qps_mid1 = get_max_qps_of_M(results, M_mid1, recall_min)
            qps_mid2 = get_max_qps_of_M(results, M_mid2, recall_min)
            if qps_mid1 <= qps_mid2:
                M_left = M_mid1
            else:
                M_right = M_mid2
    except TimeoutError:
        print("Exploration time out")

def exploitation_phase(results, ground_truth, recall_min, exploitation_budget):
    gd = ground_truth
    targets = get_exploitation_targets(results)
    len_targets = len(targets)
    if len_targets == 0:
        return
    best_M = targets[0][1]
    Ms = sorted([M for _, M, _, _ in targets])
    efC_min, efC_max = targets[0][2][0], targets[0][3][0]
    M_min, M_max = \
        Ms[Ms.index(best_M)-1 if Ms.index(best_M) > 0 else 0], \
        Ms[Ms.index(best_M)+1 if Ms.index(best_M) < len(Ms)-1 else -1]
    searched_hp = set()
    try:
        while len(searched_hp) < (M_max - M_min + 1) * (efC_max - efC_min + 1):
            M = random.randint(M_min, M_max)
            efC = random.randint(efC_min, efC_max)
            if (M, efC) in searched_hp:
                continue
            searched_hp.add((M, efC))
            efS = gd.get_efS(M, efC, recall_min)
            if gd.tuning_time > exploitation_budget:
                raise TimeoutError("tuning time out")
            perf = gd.get(M, efC, efS)
            if (M, efC, efS) not in searched_hp:
                results.append(((M, efC, efS), (gd.tuning_time, *perf)))
    except TimeoutError:
        print("Exploitation time out")

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    random.seed(SEED)
    PHASE_THRESHOLD = 0.7
    gd = GroundTruth(impl=impl, dataset=dataset)
    results = []
    exploration_phase(results, gd, recall_min, PHASE_THRESHOLD * tuning_budget)
    print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
    print(f"TT : {tuning_budget - gd.tuning_time} / {tuning_budget}")
    exploitation_phase(results, gd, recall_min, tuning_budget - gd.tuning_time)
    print("Tuning is done!")
    return results

if __name__ == "__main__":
    # for RECALL_MIN in [0.90, 0.95, 0.99]:
    for RECALL_MIN in [0.99]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["hnswlib"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="our_solution6", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)