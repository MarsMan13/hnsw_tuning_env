import math
import random

from src.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, SEED, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX, M_MIN, M_MAX
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.our_solution import get_max_perf
from src.solutions.our_solution.utils import EfCGetter, EfSGetter, get_max_perf
from data.ground_truths.ground_truth import GroundTruth

from joblib import Memory

LOWER_MUL_FACTOR, UPPER_MUL_FACTOR = 0.5, 2.0
M_to_perf = []
searched_hp = set()
efCGetter = EfCGetter()

def exploration_phase(results, ground_truth:GroundTruth, recall_min=None, qps_min=None, exploration_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf : perf[1] if recall_min else perf[0]
    gd = ground_truth
    M_left, M_right = M_MIN, M_MAX
    try:
        while 3 < M_right - M_left:
            M_mid1 = M_left + (M_right - M_left) // 3
            M_mid2 = M_right - (M_right - M_left) // 3
            for M in [M_mid1, M_mid2]:
                # efC_left, efC_right = efCGetter.get(M)
                efC_left, efC_right = EFC_MIN, EFC_MAX
                efSGetter = EfSGetter()
                efC_iter_limit = math.ceil(math.log(EFC_MAX - EFC_MIN, 3)) // 3   #! TODO
                efC_count = 0
                max_perf = 0.0
                while 3 < efC_right - efC_left and efC_count < efC_iter_limit:
                    efC_count += 1
                    efC_mid1 = efC_left + (efC_right - efC_left) // 3
                    efC_mid2 = efC_right - (efC_right - efC_left) // 3
                    efS_mid1_min, efS_mid1_max = efSGetter.get(efC_mid1)
                    efS_mid2_min, efS_mid2_max = efSGetter.get(efC_mid2)
                    efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, qps_min, efS_min=efS_mid1_min, efS_max=efS_mid1_max)
                    if gd.tuning_time > exploration_budget:
                        # gd.rollback()
                        raise TimeoutError("tuning time out")
                    tt1 = gd.tuning_time
                    efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, qps_min, efS_min=efS_mid2_min, efS_max=efS_mid2_max)
                    if gd.tuning_time > exploration_budget:
                        # gd.rollback()
                        raise TimeoutError("tuning time out")
                    tt2 = gd.tuning_time
                    efSGetter.put(efC_mid1, efS_mid1)
                    efSGetter.put(efC_mid2, efS_mid2)
                    perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
                    perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
                    if (M, efC_mid1, efS_mid1) not in searched_hp:
                        searched_hp.add((M, efC_mid1, efS_mid1))
                        results.append(((M, efC_mid1, efS_mid1), (tt1, *perf_mid1)))
                    if (M, efC_mid2, efS_mid2) not in searched_hp:
                        searched_hp.add((M, efC_mid2, efS_mid2))
                        results.append(((M, efC_mid2, efS_mid2), (tt2, *perf_mid2)))
                    if get_perf(perf_mid1) <= get_perf(perf_mid2):
                        efC_left = efC_mid1
                    else:
                        efC_right = efC_mid2
                    max(max_perf, get_perf(perf_mid1), get_perf(perf_mid2))
                    print(f"\t[{M}] - efC : {efC_mid1} -> {get_perf(perf_mid1)}, efC : {efC_mid2} -> {get_perf(perf_mid2)}")
                efCGetter.put(M, efC_left * LOWER_MUL_FACTOR, efC_right * UPPER_MUL_FACTOR)
                M_to_perf.append((M, max_perf, efC_left, efC_right))
            perf_mid1 = get_max_perf(results, M_mid1, recall_min=recall_min, qps_min=qps_min)
            perf_mid2 = get_max_perf(results, M_mid2, recall_min=recall_min, qps_min=qps_min)
            print(f"M : {M_mid1} -> {perf_mid1}, M : {M_mid2} -> {perf_mid2}\n")
            if perf_mid1 <= perf_mid2:
                M_left = M_mid1
            else:
                M_right = M_mid2
    except TimeoutError:
        print("Explration Time Out!")

def exploitation_phase(results, ground_truth:GroundTruth, exploitation_budget, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = ground_truth
    M_to_perf.sort(key=lambda x: x[1], reverse=True)
    try:
        for M, _, efC_left, efC_right in M_to_perf:
            efSGetter = EfSGetter()
            while 3 < efC_right - efC_left:
                efC_mid1 = efC_left + (efC_right - efC_left) // 3
                efC_mid2 = efC_right - (efC_right - efC_left) // 3
                efS_mid1_min, efS_mid1_max = efSGetter.get(efC_mid1)
                efS_mid2_min, efS_mid2_max = efSGetter.get(efC_mid2)
                
                efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, qps_min, efS_min=efS_mid1_min, efS_max=efS_mid1_max)
                if gd.tuning_time > exploitation_budget:
                    # gd.rollback()
                    raise TimeoutError("tuning time out")
                tt1 = gd.tuning_time
                efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, qps_min, efS_min=efS_mid2_min, efS_max=efS_mid2_max)
                if gd.tuning_time > exploitation_budget:
                    # gd.rollback()
                    raise TimeoutError("tuning time out")
                tt2 = gd.tuning_time
                efSGetter.put(efC_mid1, efS_mid1)
                efSGetter.put(efC_mid2, efS_mid2)
                
                perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
                perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
                if (M, efC_mid1, efS_mid1) not in searched_hp:
                    searched_hp.add((M, efC_mid1, efS_mid1))
                    results.append(((M, efC_mid1, efS_mid1), (tt1, *perf_mid1)))
                if (M, efC_mid2, efS_mid2) not in searched_hp:
                    searched_hp.add((M, efC_mid2, efS_mid2))
                    results.append(((M, efC_mid2, efS_mid2), (tt2, *perf_mid2)))
                if perf_mid1[1] <= perf_mid2[1]:
                    efC_left = efC_mid1
                else:
                    efC_right = efC_mid2
    except TimeoutError:
        print("Exploitation Time Out!")
    searched_M = list(set(m for (m, *_), _ in results))
    searched_M.sort()

memory = Memory("/tmp/our_solution_cache", verbose=0)
@memory.cache
def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    random.seed(SEED)
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    results = []
    exploration_phase(results, gd, recall_min, qps_min, tuning_budget)
    exploitation_tuning_budget = tuning_budget - gd.tuning_time
    if exploitation_tuning_budget > 0:
        exploitation_phase(results, gd, exploitation_tuning_budget, recall_min, qps_min)
    # print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
    print("Tuning is done!")
    
    return results

if __name__ == "__main__":
    # for RECALL_MIN in [0.90, 0.95, 0.975]:
    for RECALL_MIN in [0.90, 0.95]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular",]:
            # for DATASET in ["sift-128-euclidean"]:
                results = run(IMPL, DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)
                # print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="our_solution", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)