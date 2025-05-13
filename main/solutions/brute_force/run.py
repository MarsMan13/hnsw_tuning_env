from main.constants import DATASET, IMPL, SEED, TUNING_BUDGET
from main.utils import get_local_optimal_hyperparameter, get_optimal_hyperparameter, plot_efS_3d, plot_searched_points_3d, plot_timestamp, save_search_results
from static.ground_truths import GroundTruth

MIN_RECALL = 0.90

def run():
    gd = GroundTruth(impl=IMPL, dataset=DATASET)
    results = []
    for M in range(4, 64+1, 2):
        for efC in range(16, 512+1, 16):
            efS = gd.get_efS(M, efC, MIN_RECALL, method="binary")
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            gd.tuning_time += 100.0
            results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
            # print(f"{recall:.4f},{qps:.4f}")
    opt_hp = get_optimal_hyperparameter(results, min_recall=MIN_RECALL)
    print("Optimal Hyperparameter:")
    print(f"{opt_hp[0]} | {opt_hp[1]}")
    print("Local optimal Hyperparameters:")
    local_opt_hps = get_local_optimal_hyperparameter(results, min_recall=MIN_RECALL)
    for hp, perf in local_opt_hps:
        print(f"{hp} | {perf}")    
    return results
    
if __name__ == "__main__":
    # for IMPL in ["hnswlib", "faiss"]:
    for IMPL in ["hnswlib"]:
        # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
        for DATASET in ["nytimes-256-angular"]:
            results = run()
            save_search_results(results, solution="brute_force", filename=f"brute_force_{IMPL}_{DATASET}_{MIN_RECALL}r.csv")
            plot_timestamp(results, solution="brute_force", filename=f"brute_force_{IMPL}_{DATASET}_{MIN_RECALL}r.png", min_recall=MIN_RECALL, tuning_budget=10**20)
            plot_searched_points_3d(results, solution="brute_foce", filename=f"brute_foce_searched_points_3d_{IMPL}_{DATASET}_{MIN_RECALL}.png", min_recall=MIN_RECALL, tuning_budget=10**20)
            plot_efS_3d(results, solution="brute_foce", filename=f"brute_foce_searched_points_3d_{IMPL}_{DATASET}_{MIN_RECALL}.png", min_recall=MIN_RECALL, tuning_budget=10**20)