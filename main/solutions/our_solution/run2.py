from main.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX, M_MIN, M_MAX
from main.solutions import postprocess_results, print_optimal_hyperparameters
from static.ground_truths.ground_truth import GroundTruth

def run(impl=IMPL, dataset=DATASET, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    gd = GroundTruth(impl=impl, dataset=dataset)
    results = []
    M_step = 4
    efC_max = EFC_MAX 
    efS_min, efS_max = EFS_MIN, EFS_MAX
    # for M in range(M_MIN, M_MAX+1, M_step):
    for M in range(48, 48+1, M_step):
        if gd.tuning_time > tuning_budget:
            print(f"Tuning time out")
            break
        efC_left, efC_right = max(EFC_MIN, M), min(EFC_MAX, efC_max)
        efS_min, efS_max = EFS_MIN, min(EFS_MAX, efS_max+6400)
        while efC_right - efC_left > 3:
            efC_mid1 = efC_left + (efC_right - efC_left) // 3
            efC_mid2 = efC_right - (efC_right - efC_left) // 3
            print(efC_left, efC_mid1, efC_mid2, efC_right)
            efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=efS_min, efS_max=efS_max)
            # efS_mid2 = gd.get_efS(M, efC_mid2, recall_min, efS_min=EFS_MIN, efS_max=EFS_MAX)
            if gd.tuning_time > tuning_budget:  break
            efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=efS_min, efS_max=efS_mid2)
            # efS_mid1 = gd.get_efS(M, efC_mid1, recall_min, efS_min=EFS_MIN, efS_max=EFS_MAX)
            if gd.tuning_time > tuning_budget:  break
            
            perf_mid1 = gd.get(M, efC_mid1, efS_mid1)
            perf_mid2 = gd.get(M, efC_mid2, efS_mid2)
            print(f"{perf_mid1}\n{perf_mid2}")
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
        # efC_max = efC_right+100
    return results

if __name__ == "__main__":
    for RECALL_MIN in [0.90]:
        # for IMPL in ["hnswlib", "faiss"]:
        for IMPL in ["faiss"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular"]:
                results = run(IMPL, DATASET, RECALL_MIN, TUNING_BUDGET)
                print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                postprocess_results(
                    results, solution="the_solution", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET)