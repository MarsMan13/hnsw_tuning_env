
from main.utils import get_local_optimal_hyperparameter, get_optimal_hyperparameter
from results.postprocess import MIN_RECALL


def print_optimal_hyperparameters(results, recall_min=MIN_RECALL):
    opt_hp = get_optimal_hyperparameter(results, min_recall=recall_min)
    print("Optimal Hyperparameter:")
    print(f"{opt_hp[0]} | {opt_hp[1]}")
    print("Local optimal Hyperparameters:")
    local_opt_hps = get_local_optimal_hyperparameter(results, min_recall=recall_min)
    for hp, perf in local_opt_hps:
        print(f"{hp} | {perf}")    