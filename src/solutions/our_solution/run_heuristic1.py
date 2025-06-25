import math
import random

from src.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, SEED, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX, M_MIN, M_MAX
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.our_solution.utils import EfCGetter, EfSGetter, get_max_perf
from data.ground_truths.ground_truth import GroundTruth
from functools import cmp_to_key
from joblib import Memory

# --- Global variables for tracking search state ---
_M_to_perf = [] # [(M, max_perf, efC_left, efC_right), ...]
_searched_hp = set()
_efC_getter = EfCGetter()
_efS_getters = dict()

def _find_best_efc_for_m(M, ground_truth:GroundTruth, results, get_perf, recall_min, qps_min, exploration_budget):
    
    # efC_left, efC_right = EFC_MIN, EFC_MAX
    efC_left, efC_right = _efC_getter.get(M)
    # efC_left, efC_right = max(0.9 * efC_left, EFC_MIN), min(1.1 * efC_right, EFC_MAX)
    if M not in _efC_getter:
        _efS_getters[M] = EfSGetter()
    efS_getter = _efS_getters[M]
    
    efC_iter_limit = math.ceil(math.log(EFC_MAX - EFC_MIN, 3)) // 2     # ! Logic preserved as per request.
    # efC_iter_limit = math.ceil(math.log(EFC_MAX - EFC_MIN, 2.5)) // 2 # ! Logic preserved as per request.
    efC_count = 0
    max_perf_of_M = 0.0
    while 3 < efC_right - efC_left and efC_count < efC_iter_limit:
        efC_count += 1
        
        efC_mid1 = efC_left + (efC_right - efC_left) // 3
        efC_mid2 = efC_right - (efC_right - efC_left) // 3
        
        efS_mid1_min, efS_mid1_max = efS_getter.get(efC_mid1)
        efS_mid2_min, efS_mid2_max = efS_getter.get(efC_mid2)
        efS_mid1 = ground_truth.get_efS(M, efC_mid1, recall_min, qps_min, efS_min=efS_mid1_min, efS_max=efS_mid1_max)

        if ground_truth.tuning_time > exploration_budget:
            raise TimeoutError("Tuning time out during efS search")
        tt1 = ground_truth.tuning_time

        efS_mid2 = ground_truth.get_efS(M, efC_mid2, recall_min, qps_min, efS_min=efS_mid2_min, efS_max=efS_mid2_max)
        if ground_truth.tuning_time > exploration_budget:
            raise TimeoutError("Tuning time out during efS search")
        tt2 = ground_truth.tuning_time
        
        efS_getter.put(efC_mid1, efS_mid1)
        efS_getter.put(efC_mid2, efS_mid2)
        perf_mid1 = ground_truth.get(M, efC_mid1, efS_mid1)
        perf_mid2 = ground_truth.get(M, efC_mid2, efS_mid2)

        hp1 = (M, efC_mid1, efS_mid1)
        if hp1 not in _searched_hp:
            _searched_hp.add(hp1)
            results.append((hp1, (tt1, *perf_mid1)))

        hp2 = (M, efC_mid2, efS_mid2)
        if hp2 not in _searched_hp:
            _searched_hp.add(hp2)
            results.append((hp2, (tt2, *perf_mid2)))

        if get_perf(perf_mid1) <= get_perf(perf_mid2):
            efC_left = efC_mid1
        else:
            efC_right = efC_mid2
        
        max_perf_of_M = max(max_perf_of_M, get_perf(perf_mid1), get_perf(perf_mid2))
        # print(f"\t\t[{M}] - efC: {efC_mid1} -> {get_perf(perf_mid1):.4f}, efC: {efC_mid2} \
        #         -> {get_perf(perf_mid2):.4f} -> Max Perf for M: {max_perf_of_M:.4f}")

    _efC_getter.put(M, efC_left, efC_right)
    _M_to_perf.append((M, max_perf_of_M, efC_left, efC_right))
    # print(f"\tM : {M} -> {max_perf_of_M:.4f}")
    return max_perf_of_M

def _exploration_phase(results, ground_truth: GroundTruth, recall_min=None, qps_min=None, exploration_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    
    get_perf = lambda perf: perf[1] if recall_min else perf[0]
    M_left, M_right = M_MIN, M_MAX
    _M_to_perf = []
    processed_M = set([m for m, *_ in _M_to_perf])

    try:
        while 3 < M_right - M_left:
            M_mid1 = M_left + (M_right - M_left) // 3
            M_mid2 = M_right - (M_right - M_left) // 3

            for M in [M_mid1, M_mid2]:
                _find_best_efc_for_m(M, ground_truth, results, get_perf, recall_min, qps_min, exploration_budget)
                processed_M.add(M)

            perf_mid1 = get_max_perf(results, M_mid1, recall_min=recall_min, qps_min=qps_min)
            perf_mid2 = get_max_perf(results, M_mid2, recall_min=recall_min, qps_min=qps_min)
            # print(f"M : {M_mid1} -> {perf_mid1}, M : {M_mid2} -> {perf_mid2}\n")
            
            _M_to_perf.append((M_mid1, perf_mid1, M_left, M_right))
            _M_to_perf.append((M_mid2, perf_mid2, M_left, M_right))

            if perf_mid1 <= perf_mid2:
                M_left = M_mid1
            else:
                M_right = M_mid2

        for M in [M_left, M_right]:
            if M in processed_M:
                continue
            
            max_perf_of_M = _find_best_efc_for_m(M, ground_truth, results, get_perf, exploration_budget)
            processed_M.add(M)
            _M_to_perf.append((M, max_perf_of_M, M_left, M_right))
    except TimeoutError as e:
        print(f"Exploration Time Out! Details: {e}")
    _M_to_perf = _M_to_perf[::-1]
    for M, perf, M_left_debug, M_right_debug in _M_to_perf:
        print(f"{M} {round(float(perf))} {M_left_debug} {M_right_debug}")

def _exploitation_phase(results, ground_truth:GroundTruth, exploitation_budget, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = ground_truth
    get_perf = lambda perf: perf[1] if recall_min else perf[0]
    if not _M_to_perf:
        print("No M to perf found in exploration phase.")
        return
    sorted_M_configs = sorted(_M_to_perf, key=lambda x: x[1], reverse=True)
    print
    try:
        for M, max_perf, efC_left, efC_right in sorted_M_configs:
            if gd.tuning_time > exploitation_budget:
                print("Exploitation budget exceeded!")
                break
            _find_best_efc_for_m(M, gd, results, get_perf, recall_min, qps_min, exploitation_budget)
        print(f"All M values explored in exploitation phase: {len(sorted_M_configs)}")
    except TimeoutError:
        print("Exploitation Time Out!")
    searched_M = list(set(m for (m, *_), _ in results))
    searched_M.sort()

# memory = Memory("/tmp/our_solution_cache", verbose=0)
# @memory.cache
def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    random.seed(SEED)
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    _M_to_perf.clear()
    _searched_hp.clear()
    _efC_getter.clear()
    _efS_getters.clear()
    results = []
    _exploration_phase(results, gd, recall_min, qps_min, tuning_budget)
    print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
    exploitation_tuning_budget = tuning_budget - gd.tuning_time
    if exploitation_tuning_budget > 0:
        print(f"Exploitation tuning budget: {exploitation_tuning_budget}")
        _exploitation_phase(results, gd, tuning_budget, recall_min, qps_min)
        print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
    print("Tuning is done!")
    
    return results

if __name__ == "__main__":
    for RECALL_MIN in [0.95]:
    # for RECALL_MIN in [0.90, 0.95, 0.975]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["hnswlib"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "youtube-1024-angular"]:
            for DATASET in ["dbpediaentity-768-angular"]:
                print(f"Running for {IMPL} on {DATASET} with RECALL_MIN={RECALL_MIN}")
                results = run(IMPL, DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)
                # print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="our_solution", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)