import math
from unittest import skip

from numpy import append

from main.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN, M_MIN, M_MAX, EFC_MIN, EFC_MAX,EFS_MIN, EFS_MAX 
from main.solutions import postprocess_results, print_optimal_hyperparameters
from main.solutions.our_solution import get_max_qps_of_M, get_exploitation_targets, find_value_of_dict
from main.solutions.our_solution.exploitations import exploitation_top_k_M
from static.ground_truths.ground_truth import GroundTruth

def exploration_phase(results, ground_truth, recall_min, exploration_budget):
    gd = ground_truth 
    efC_left, efC_right = EFC_MIN, EFC_MAX 
    efS_min, efS_max = EFS_MIN, EFS_MAX
    M_left, M_right = M_MIN, M_MAX
    searched_hp = set()
    M_efC = dict()  # M -> (efC_min, efC_max) of the best performance
    M_efS = dict()  # M -> (efS_min, efC_max) of the best performance
    try:
        while M_right - M_left > 3:
            M_mid1 = M_left + (M_right - M_left) // 3
            M_mid2 = M_right - (M_right - M_left) // 3
            for M in [M_mid1, M_mid2]:
                efC_left = find_value_of_dict(M_efC, M, mode="min")[0]
                efC_left = max(EFC_MIN, M, efC_left if efC_left is not None else EFC_MIN)
                efC_right = find_value_of_dict(M_efC, M, mode="max")[1]
                efC_right = min(EFC_MAX, efC_right if efC_right is not None else EFC_MAX)
                # print(f"{M:2} {efC_left:3} {efC_right:3}")
                # efS_max = min(int(efS_max * 1.1), EFS_MAX)    #! TODO
                # efS_local_max = EFS_MAX 
                # max_efC_iter = int((EFC_MAX - EFC_MIN) * (3/2) ** math.ceil(math.log(EFC_MAX - EFC_MIN, 1.5))) // 3   #! TODO : Hyperparameter
                max_efC_iter = 1000
                efC_gap = int((EFC_MAX - EFC_MIN) * ((2/3) ** ((math.ceil(math.log(EFC_MAX - EFC_MIN, 1.5))) // 3)))    #! TODO : Hyperparameter
                efC_count = 0
                while efC_right - efC_left > 3 and efC_count < max_efC_iter and efC_right - efC_left > efC_gap:
                    efC_count += 1
                    efC_mid1 = efC_left + (efC_right - efC_left) // 3
                    efC_mid2 = efC_right - (efC_right - efC_left) // 3
                    # print(f"   {M:2} | {efC_left:3} {efC_mid1:3} {efC_mid2:3} {efC_right:3}")
                    efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=efS_min, efS_max=efS_max)
                    efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=efS_min, efS_max=efS_mid2, skip_time=True)
                    if gd.tuning_time > exploration_budget:
                        gd.rollback() 
                        raise TimeoutError("tuning time out")
                    perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
                    perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
                    if (M, efC_mid1, efS_mid1) not in searched_hp:
                        results.append(((M, efC_mid1, efS_mid1), (gd.tuning_time, *perf_mid1)))
                        searched_hp.add((M, efC_mid1, efS_mid1))
                    if (M, efC_mid2, efS_mid2) not in searched_hp:
                        results.append(((M, efC_mid2, efS_mid2), (gd.tuning_time, *perf_mid2)))
                        searched_hp.add((M, efC_mid2, efS_mid2))
                    if perf_mid1[1] <= perf_mid2[1]:
                        efC_left = efC_mid1
                        efS_min = efS_mid1
                    else:
                        efC_right = efC_mid2
                        efS_max = efS_mid2
                M_efC[M] = (int(efC_left * 0.5), int(efC_right * 1.5))    #! TODO : Check it
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
    if len_targets == 0:    # No targets found
        return
    appended_hp = set([(M, efC, efS) for (M, efC, efS), _ in results])
    try:
        for alloc, M, first, second in targets:
            efC_left = min(first[0], second[0])
            efC_right = max(first[0], second[0])
            efS_local_min = min(first[2], second[2])
            efS_local_max = max(first[2], second[2])
            # print(f"{M} | {efC_left} {efC_right}")
            while efC_right - efC_left > 3:
                efC_mid1 = efC_left + (efC_right - efC_left) // 3
                efC_mid2 = efC_right - (efC_right - efC_left) // 3
                efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=efS_local_min, efS_max=efS_local_max)
                efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=efS_local_min, efS_max=efS_mid2)
                if gd.tuning_time > exploitation_budget:
                    raise TimeoutError("tuning time out")
                perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
                perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
                # print(f"\t{M:3} | {efC_left} {efC_mid1} {efC_mid2} {efC_right} ({gd.tuning_time})")
                # print(f"\tQPS | {perf_mid1[1]} {perf_mid2[1]}")
                if (M, efC_mid1, efS_mid1) not in appended_hp:
                    results.append(((M, efC_mid1, efS_mid1), (gd.tuning_time, *perf_mid1)))
                    appended_hp.add((M, efC_mid1, efS_mid1))
                if (M, efC_mid2, efS_mid2) not in appended_hp:
                    results.append(((M, efC_mid2, efS_mid2), (gd.tuning_time, *perf_mid2)))
                    appended_hp.add((M, efC_mid2, efS_mid2))
                if perf_mid1[1] <= perf_mid2[1]:
                    efC_left = efC_mid1
                    efS_local_min = efS_mid1
                else:
                    efC_right = efC_mid2
                    efS_local_max = efS_mid2
                if gd.tuning_time > exploitation_budget * alloc:    #! TODO
                # if gd.tuning_time > exploitation_budget:
                    break
            if gd.tuning_time > exploitation_budget * alloc:
            # if gd.tuning_time > exploitation_budget:
                continue
    except TimeoutError:
        print("Exploitation time out")

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    PHASE_THRESHOLD = 0.7
    gd = GroundTruth(impl=impl, dataset=dataset)
    results = []
    exploration_phase(results, gd, recall_min, PHASE_THRESHOLD * tuning_budget)
    print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
    print(f"t/T : {gd.tuning_time / tuning_budget * 100}")
    if gd.tuning_time < tuning_budget:
        exploitation_top_k_M(results, gd, recall_min, tuning_budget - gd.tuning_time)
    #     exploitation_phase(results, gd, recall_min, tuning_budget - gd.tuning_time)
    print("Tuning is done!")
    return results

if __name__ == "__main__":
    # for RECALL_MIN in [0.90, 0.95, 0.99]:
    for RECALL_MIN in [0.90]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["hnswlib"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["sift-128-euclidean"]:
                print(f"{'#' * 20} {IMPL} | {DATASET} | {RECALL_MIN} {'#' * 20}")
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="our_solution5", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)