import math
import random

from numpy import append

from main.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, SEED, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX, M_MIN, M_MAX
from main.solutions import postprocess_results, print_optimal_hyperparameters
from main.solutions.our_solution import get_max_qps_of_M, get_exploitation_targets
from main.solutions.our_solution.utils import MToEf
from static.ground_truths.ground_truth import GroundTruth

def exploration_phase(results, ground_truth, recall_min, exploration_budget):
    gd = ground_truth 
    # efC_left, efC_right = EFC_MIN, EFC_MAX 
    # efS_min, efS_max = EFS_MIN, EFS_MAX
    M_left, M_right = M_MIN, M_MAX
    searched_hp = set()
    M_to_efC = MToEf()
    M_to_efS = MToEf()
    try:
        while M_right - M_left > 3: # FOR Ms
            M_mid1 = M_left + (M_right - M_left) // 3
            M_mid2 = M_right - (M_right - M_left) // 3
            for M in [M_mid1, M_mid2]:
                # Heuristic 2-1
                efC_left, efC_right = M_to_efC.get_range(M)
                efC_left, efC_right = max(int(0.9 * efC_left), EFC_MIN), min(int(1.1 * efC_right), EFC_MAX)
                efS_left, efS_right = M_to_efS.get_range(M)
                efS_left, efS_right = max(int(0.9 * efS_left), EFS_MIN), min(int(1.1 * efS_right), EFS_MAX)
                print(f"{M:2} | {efC_left:3} {efC_right:3}")
                efC_iter_limit = math.ceil(math.log(EFC_MAX - EFC_MIN, 2.5)) // 2   #! TODO
                efC_count = 0
                # Heuristic 3
                while efC_right - efC_left > 3 and efC_count < efC_iter_limit:
                    efC_count += 1
                    efC_mid1 = efC_left + (efC_right - efC_left) // 3
                    efC_mid2 = efC_right - (efC_right - efC_left) // 3
                    print(f"\t{M:2} | {efC_left:3} <= {efC_mid1:3} <= {efC_mid2:3} <= {efC_right:3}")
                    efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=efS_left, efS_max=efS_right)
                    if gd.tuning_time > exploration_budget:  
                        gd.rollback()
                        raise TimeoutError("tuning time out")
                    tt1 = gd.tuning_time
                    efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=efS_left, efS_max=efS_mid2)
                    if gd.tuning_time > exploration_budget:  
                        gd.rollback()
                        raise TimeoutError("tuning time out")
                    tt2 = gd.tuning_time
                    perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
                    perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
                    if (M, efC_mid1, efS_mid1) not in searched_hp:
                        searched_hp.add((M, efC_mid1, efS_mid1))
                        results.append(((M, efC_mid1, efS_mid1), (tt1, *perf_mid1)))
                    if (M, efC_mid2, efS_mid2) not in searched_hp:
                        results.append(((M, efC_mid2, efS_mid2), (tt2, *perf_mid2)))
                        searched_hp.add((M, efC_mid2, efS_mid2))
                    if perf_mid1[1] <= perf_mid2[1]:
                        efC_left = efC_mid1
                        efS_left = efS_mid1
                    else:
                        efC_right = efC_mid2
                        efS_right = efS_mid2
                # Heuristic 2-2
                M_to_efC.put(M, efC_left, efC_right)
                # Heuristic 3-2
                M_to_efS.put(M, efS_left, efS_right)
            # Heuristic 1
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
    PHASE_THRESHOLD = 1.0
    gd = GroundTruth(impl=impl, dataset=dataset)
    results = []
    exploration_phase(results, gd, recall_min, PHASE_THRESHOLD * tuning_budget)
    print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
    print(f"TT : {tuning_budget - gd.tuning_time} / {tuning_budget}")
    # exploitation_phase(results, gd, recall_min, tuning_budget - gd.tuning_time)
    print("Tuning is done!")
    return results

if __name__ == "__main__":
    for RECALL_MIN in [0.90, 0.95, 0.975]:
    # for RECALL_MIN in [0.95]:
        for IMPL in ["hnswlib", "faiss"]:
        # for IMPL in ["hnswlib"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="our_solution7", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)