from re import M
from main.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX, M_MIN, M_MAX
from main.solutions import postprocess_results, print_optimal_hyperparameters
from static.ground_truths.ground_truth import GroundTruth

def get_max_qps_of_M(results, M, recall_min):
    max_qps = 0
    for result in results:
        if result[0][0] == M and result[1][1] >= recall_min:
            max_qps = max(max_qps, result[1][2])
    return max_qps


def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl=impl, dataset=dataset)
    results = []
    efC_max = EFC_MAX 
    efS_min, efS_max = EFS_MIN, EFS_MAX
    M_left, M_right = M_MIN, M_MAX
    try:
        while M_right - M_left > 3:
            M_mid1 = M_left + (M_right - M_left) // 3
            M_mid2 = M_right - (M_right - M_left) // 3
            for M in [M_mid1, M_mid2]:
                efC_left, efC_right = max(EFC_MIN, M), min(EFC_MAX, efC_max)
                efS_min, efS_max = EFS_MIN, min(EFS_MAX, efS_max+6400)  #TODO : efS_max
                # efC_count = 0
                while efC_right - efC_left > 3:
                    # efC_count += 1
                    # print(f"\t[{efC_count}]efC_gap : {efC_right - efC_left}")
                    efC_mid1 = efC_left + (efC_right - efC_left) // 3
                    efC_mid2 = efC_right - (efC_right - efC_left) // 3
                    # print(M, ":", efC_left, efC_mid1, efC_mid2, efC_right)
                    # efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=EFS_MIN, efS_max=EFS_MAX)
                    # efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=EFS_MIN, efS_max=EFS_MAX)
                    efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=efS_min, efS_max=efS_max)
                    efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=efS_min, efS_max=efS_mid2)
                    if gd.tuning_time > tuning_budget:  
                        raise TimeoutError("tuning time out")
                    # Below get is not consuming time due to the cache 
                    perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
                    perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
                    #// print(f"{perf_mid1}\n{perf_mid2}")
                    results.append(
                        ((M, efC_mid1, efS_mid1), 
                        (gd.tuning_time, *perf_mid1))
                    )
                    results.append(
                        ((M, efC_mid2, efS_mid2), 
                        (gd.tuning_time, *perf_mid2))
                    )
                    if perf_mid1[1] <= perf_mid2[1]:
                        efC_left = efC_mid1
                        efS_min = efS_mid1
                    else:
                        efC_right = efC_mid2
                        efS_max = efS_mid2
                # print(f"efC_count: {efC_count}")
            if get_max_qps_of_M(results, M_mid1, recall_min) \
                <= get_max_qps_of_M(results, M_mid2, recall_min):
                M_left = M_mid1
            else:
                M_right = M_mid2        # efC_max = efC_right+100
    except TimeoutError as e:
        print("Tuning time out")
    return results

if __name__ == "__main__":
    for RECALL_MIN in [0.90]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
                            "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            # for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="our_solution3", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)