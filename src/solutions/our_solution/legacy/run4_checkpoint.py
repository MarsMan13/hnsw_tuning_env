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

def _find_best_efc_for_m(M, ground_truth:GroundTruth, results, get_perf, recall_min, qps_min, tuning_budget, is_exploitation=False):
    # efC_left, efC_right = EFC_MIN, EFC_MAX
    # efC_left, efC_right = max(0.9 * efC_left, EFC_MIN), min(1.1 * efC_right, EFC_MAX)
    efC_left, efC_right = _efC_getter.get(M)
    if M not in _efC_getter:
        _efS_getters[M] = EfSGetter()
    efS_getter = _efS_getters[M]
    efC_iter_limit = math.ceil(math.log(EFC_MAX - EFC_MIN, 2.5)) // 3 if not is_exploitation else EFC_MAX  #! HP
    max_perf_of_M = 0.0
    efC_count = 0
    while 3 < efC_right - efC_left and efC_count < efC_iter_limit:
        efC_count += 1
        
        efC_mid1 = efC_left + (efC_right - efC_left) // 3
        efC_mid2 = efC_right - (efC_right - efC_left) // 3
        print(f"\n\t\tefC : ({efC_left}) - ({efC_mid1}) - ({efC_mid2}) - ({efC_right})")
        
        efS_mid1_min, efS_mid1_max = efS_getter.get(efC_mid1)
        efS_mid2_min, efS_mid2_max = efS_getter.get(efC_mid2)
        
        efS_mid1 = ground_truth.get_efS(M, efC_mid1, recall_min, qps_min, efS_min=efS_mid1_min, efS_max=efS_mid1_max)
        if ground_truth.tuning_time > tuning_budget:
            break
        tt1 = ground_truth.tuning_time
        efS_getter.put(efC_mid1, efS_mid1)
        hp1 = (M, efC_mid1, efS_mid1)
        perf_mid1 = ground_truth.get(M, efC_mid1, efS_mid1)
        if hp1 not in _searched_hp:
            _searched_hp.add(hp1)
            results.append((hp1, (tt1, *perf_mid1)))
        
        efS_mid2 = ground_truth.get_efS(M, efC_mid2, recall_min, qps_min, efS_min=efS_mid2_min, efS_max=efS_mid2_max)
        if ground_truth.tuning_time > tuning_budget:
            break
        tt2 = ground_truth.tuning_time
        efS_getter.put(efC_mid2, efS_mid2)
        hp2 = (M, efC_mid2, efS_mid2)
        perf_mid2 = ground_truth.get(M, efC_mid2, efS_mid2)
        if hp2 not in _searched_hp:
            _searched_hp.add(hp2)
            results.append((hp2, (tt2, *perf_mid2)))
        
        perf1 = get_perf(perf_mid1)
        perf2 = get_perf(perf_mid2)
        if perf1 == perf2:  #* It includes the case when perf1 == perf2 == 0.0
            if recall_min is not None:
                efC_left = efC_mid1
            else:
                efC_right = efC_mid2
            efC_count -= 1
        elif perf1 <= perf2:
            efC_left = efC_mid1
        else:
            efC_right = efC_mid2
        _efC_getter.put(M, efC_left, efC_right) #! <- Check this line, it might make bad effects on the search space.
        max_perf_of_M = max(max_perf_of_M, perf1, perf2)
        print(f"\t\t[{M}] - efC: {efC_mid1} -> {get_perf(perf_mid1):.4f}, efC: {efC_mid2} -> {get_perf(perf_mid2):.4f} -> Max Perf for M: {max_perf_of_M:.4f}")

    for efC in range(efC_left, efC_right + 1):
        if efC_iter_limit <= efC_count:
            break
        if efC in efS_getter:
            continue
        efC_count += 1
        efS_min, efS_max = efS_getter.get(efC)
        efS = ground_truth.get_efS(M, efC, recall_min, qps_min, efS_min=efS_min, efS_max=efS_max)
        if ground_truth.tuning_time > tuning_budget:
            break
        tt = ground_truth.tuning_time
        efS_getter.put(efC, efS)
        hp = (M, efC, efS)
        perf = ground_truth.get(M, efC, efS)
        if hp not in _searched_hp:
            _searched_hp.add(hp)
            results.append((hp, (tt, *perf)))
        _efC_getter.put(M, efC, efC_right)  # Ensure efC_right is updated
        max_perf_of_M = max(max_perf_of_M, get_perf(perf))
    print(f"\tM : {M} -> {max_perf_of_M:.4f}")
    return max_perf_of_M

def _exploration_phase(results, ground_truth: GroundTruth, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf: perf[1] if recall_min else perf[0]
    _M_to_perf.clear() # [(M, max_perf)]
    M_left, M_right = M_MIN, M_MAX
    processed_M = set([m for m, *_ in _M_to_perf])
    try:
        while 3 < M_right - M_left:
            M_mid1 = M_left + (M_right - M_left) // 3
            M_mid2 = M_right - (M_right - M_left) // 3

            perf_mid1 = _find_best_efc_for_m(M_mid1, ground_truth, results, get_perf, recall_min, qps_min, tuning_budget)
            if ground_truth.tuning_time > tuning_budget:
                raise TimeoutError("Exploration Tuning time out")
            _M_to_perf.append((M_mid1, perf_mid1))
            processed_M.add(M_mid1)
            
            perf_mid2 = _find_best_efc_for_m(M_mid2, ground_truth, results, get_perf, recall_min, qps_min, tuning_budget)
            if ground_truth.tuning_time > tuning_budget:
                raise TimeoutError("Exploration Tuning time out")
            _M_to_perf.append((M_mid2, perf_mid2))
            processed_M.add(M_mid2)

            print(f"{M_left} < {M_mid1} -> {perf_mid1} < M : {M_mid2} -> {perf_mid2} < {M_right}\n")
            if perf_mid1 == perf_mid2:
                if recall_min is not None:
                    M_left = M_mid1
                else:
                    M_right = M_mid2
            else:
                perf_mid1 = perf_mid1 * 0.95 if recall_min else perf_mid1
                if perf_mid1 <= perf_mid2:
                    M_left = M_mid1
                else:
                    M_right = M_mid2

        for M in range(M_left, M_right+1):
            if M in processed_M:
                continue
            perf = _find_best_efc_for_m(M, ground_truth, results, get_perf, recall_min, qps_min, tuning_budget)
            if ground_truth.tuning_time > tuning_budget:
                raise TimeoutError("Exploitation Tuning time out")
            _M_to_perf.append((M, perf))
            processed_M.add(M)
    except TimeoutError as e:
        print(f"Tuning Time Out!")
    #* Debugging output ====
    # _M_to_perf = _M_to_perf[::-1]
    # for M, perf, M_left_debug, M_right_debug in _M_to_perf:
    #     print(f"{M} {round(float(perf))} {M_left_debug} {M_right_debug}")

def _exploitation_phase(results, ground_truth:GroundTruth, tuning_budget, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf: perf[1] if recall_min else perf[0]
    if not _M_to_perf:
        print("No M to perf found in exploration phase.")
        return
    sorted_M_configs = sorted(_M_to_perf, key=lambda x: x[1], reverse=True)
    try:
        for M, max_perf in sorted_M_configs:
            _ = _find_best_efc_for_m(M, ground_truth, results, get_perf, recall_min, qps_min, tuning_budget, is_exploitation=True)
            if ground_truth.tuning_time > tuning_budget:
                raise TimeoutError("Exploitation Tuning time out")
        print(f"All M values explored in exploitation phase: {len(sorted_M_configs)}")
    except TimeoutError:
        print("Tuning Time Out!")
    searched_M = list(set(m for (m, *_), _ in results))
    searched_M.sort()

# memory = Memory("/tmp/our_solution_cache", verbose=0)
# @memory.cache
def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    random.seed(SEED)
    ground_truth = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    _M_to_perf.clear()
    _searched_hp.clear()
    _efC_getter.clear()
    _efS_getters.clear()
    results = []
    _exploration_phase(results, ground_truth, recall_min, qps_min, tuning_budget)
    if ground_truth.tuning_time < tuning_budget:
        print(f"Exploitation tuning budget: {tuning_budget - ground_truth.tuning_time:.2f}s")
        _exploitation_phase(results, ground_truth, tuning_budget, recall_min, qps_min)
    print_optimal_hyperparameters(results, recall_min=recall_min, qps_min=qps_min)
    print("Tuning is done!")
    
    return results

def run_recall_min_experiments():    
    for RECALL_MIN in [0.90]:
    # for RECALL_MIN in [0.90, 0.95, 0.975]:
        for IMPL in ["faiss"]:
        # for IMPL in ["milvus"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular"]:
            for DATASET in ["glove-100-angular"]:
                print(f"Running for {IMPL} on {DATASET} with RECALL_MIN={RECALL_MIN}")
                results = run(IMPL, DATASET, recall_min=RECALL_MIN, qps_min=None, tuning_budget=TUNING_BUDGET)
                # print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                # postprocess_results(
                #     results, solution="our_solution", impl=IMPL, dataset=DATASET, 
                #     recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)

def run_qps_min_experiments():
    for QPS_MIN in [18268]:
        for IMPL in ["faiss"]:
            for DATASET in ["glove-100-angular"]:
            # for DATASET in ["dbpediaentity-768-angular"]:
                print(f"Running for {IMPL} on {DATASET} with QPS_MIN={QPS_MIN}")
                results = run(IMPL, DATASET, recall_min=None, qps_min=QPS_MIN, tuning_budget=TUNING_BUDGET)
                # postprocess_results(
                #     results, solution="our_solution", impl=IMPL, dataset=DATASET, 
                #     qps_min=QPS_MIN, tuning_budget=TUNING_BUDGET)

if __name__ == "__main__":
    run_recall_min_experiments()
    # run_qps_min_experiments()