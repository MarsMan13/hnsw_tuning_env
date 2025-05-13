from main.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, TUNING_BUDGET
from main.utils import plot_efS_3d, plot_searched_points_3d, plot_timestamp, save_search_results
from static.ground_truths.ground_truth import GroundTruth

RECALL_MIN = 0.90

def run():
    gd = GroundTruth(impl=IMPL, dataset=DATASET)
    results = []
    M_step = 2
    for M in range(4, 64 + 1, M_step):
        if gd.tuning_time > TUNING_BUDGET:
            print(f"Tuning time out")
            break
        efC_left, efC_right = max(8, M), 512
        efS = EFS_MAX
        opt_hp = None
        while efC_right - efC_left > 1:
            efC = (efC_left + efC_right) // 2
            result = gd.get(M, efC, EFS_MIN)
            if gd.tuning_time > TUNING_BUDGET:  break
            results.append(((M, efC, EFS_MIN), (gd.tuning_time, result[0], result[1], result[2], result[3], result[4])))
            if RECALL_MIN <= result[0]:
                efC_right = efC
            _efS = gd.get_efS(M, efC, RECALL_MIN, method="binary", efS_max=min(efS+16, EFS_MAX))    # +4 for safety
            # _efS = gd.get_efS(M, efC, RECALL_MIN, method="binary", efS_max=efS_max)    # +4 for safety
            if _efS == 0:
                efC_left = efC
                continue
            efS = _efS
            recall, qps, total_time, build_time, index_size = gd.get(M, efC, efS)
            results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
            if opt_hp is None or qps > opt_hp[1][1]:
                efC_right = efC
                opt_hp = ((M, efC, efS), (recall, qps, total_time, build_time, index_size))
            else:
                efC_left = efC
        print(opt_hp)
    return results

if __name__ == "__main__":
    results = run() 
    save_search_results(results, solution="the_solution", filename=f"the_solution_{IMPL}_{DATASET}_{RECALL_MIN}r.csv")
    plot_timestamp(results, solution="the_solution", filename=f"the_solution_{IMPL}_{DATASET}_{RECALL_MIN}r.png", min_recall=RECALL_MIN)
    plot_searched_points_3d(results, solution="the_solution", filename=f"the_solution_searched_points_3d_{IMPL}_{DATASET}_{RECALL_MIN}.png", min_recall=RECALL_MIN)
    plot_efS_3d(results, solution="the_solution", filename=f"the_solution_searched_points_3d_{IMPL}_{DATASET}_{RECALL_MIN}.png", min_recall=RECALL_MIN)