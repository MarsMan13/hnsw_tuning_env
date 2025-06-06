from main.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX
from main.solutions import postprocess_results, print_optimal_hyperparameters
from static.ground_truths.ground_truth import GroundTruth

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl=impl, dataset=dataset)
    results = []
    M_step = 2
    efC = EFC_MAX 
    local_opt = None
    for M in range(4, 64 + 1, M_step):
        if gd.tuning_time > tuning_budget:
            print(f"Tuning time out")
            break
        efC_left, efC_right = max(EFC_MIN, M), 512 if local_opt is None else local_opt[0][1]
        efS = EFS_MAX
        local_opt = None
        _count = 0
        while efC_right - efC_left > 1:
            efC = (efC_left + efC_right) // 2
            result = gd.get(M, efC, EFS_MIN)
            if gd.tuning_time > tuning_budget:  break
            results.append(
                ((M, efC, EFS_MIN), 
                (gd.tuning_time, result[0], result[1], result[2], result[3], result[4]))
            )
            if recall_min <= result[0]:
                efC_right = efC
            _efS = gd.get_efS(M, efC, recall_min, method="binary", efS_max=min(efS+16, EFS_MAX))    # +4 for safety
            # _efS = gd.get_efS(M, efC, RECALL_MIN, method="binary", efS_max=efS_max)    # +4 for safety
            if gd.tuning_time > tuning_budget:  break
            if _efS == 0:
                efC_left = efC
                continue
            efS = _efS
            recall, qps, total_time, build_time, index_size = gd.get(M, efC, efS)
            results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
            if local_opt is None or qps > local_opt[1][1]:
                efC_right = efC
                local_opt = ((M, efC, efS), (recall, qps, total_time, build_time, index_size))
            else:
                efC_left = efC
            _count += 1
        print(f"count: {_count}")
    return results

if __name__ == "__main__":
    for RECALL_MIN in [0.90]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["hnswlib"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="the_solution", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)