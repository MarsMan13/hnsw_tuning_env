from src.constants import SEED
from src.utils import get_local_optimal_hyperparameter, get_optimal_hyperparameter, plot_efS_3d, plot_multi_accumulated_timestamp, plot_searched_points_3d, plot_timestamp, save_search_results

def print_optimal_hyperparameters(results, recall_min=None, qps_min=None):
    opt_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
    print("[Optimal Hyperparameter]")
    if opt_hp is None:
        print("No optimal hyperparameter found")
        return
    print(f"{opt_hp[0]} | {opt_hp[1]}")
    print("[Local Optimal Hyperparameters]")
    local_opt_hps = get_local_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
    for hp, perf in local_opt_hps:
        print(f"{hp} | {perf}")
    print("")

def postprocess_results(results, solution, impl, dataset, tuning_budget, recall_min=None, qps_min=None, seed=SEED):
    save_search_results(
        results, 
        solution=solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.csv", seed=seed
    )
    plot_timestamp(
        results, 
        solution=solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed
    )
    plot_multi_accumulated_timestamp(
        results, 
        solution=solution, filename=f"{solution}_multi_accumulated_timestamp_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed
    )
    try:
        plot_searched_points_3d(
            results, 
            solution=solution, 
            filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
            recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed
        )
        plot_efS_3d(
            results, 
            solution=solution, 
            filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
            recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed
        )
    except Exception as e:
        print(f"Error in plotting 3D graphs: {e}")