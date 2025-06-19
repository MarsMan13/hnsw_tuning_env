import itertools

from src.constants import TUNING_BUDGET
from src.utils import filename_builder, get_optimal_hyperparameter, load_search_results, \
    plot_multi_accumulated_timestamp, plot_searched_points_3d, plot_timestamp, save_optimal_hyperparameters

# def process_file():
#     SOLUTION = "vd_tuner"
#     FILENAME = "vd_tuner_hnswlib_nytimes-256-angular_8h_False_3.csv"
#     TUNING_BUDGET = 3600 * 8
#     RECALL_MIN = 0.95
#     results = load_search_results(SOLUTION, FILENAME)
#     _FILENAME = FILENAME.split(".csv")[0]
#     plot_timestamp(results, SOLUTION, f"{_FILENAME}_timestamp_plot.png", recall_min=RECALL_MIN)
#     plot_searched_points_3d(results, SOLUTION, f"{_FILENAME}_searched_points_3d.png", recall_min=RECALL_MIN)

MOCK_SEED = 0

def _process_single_metric(
    impl: str, 
    dataset: str, 
    solutions: list, 
    recall_min: float = None, 
    qps_min: int = None
):
    """
    Helper function to process results for a single metric (either recall_min or qps_min).
    It loads data, plots timestamp, and saves optimal hyperparameters.
    """
    results_combi = {}
    optimal_combi = {}

    #* 1. Load results for all solutions under the given condition
    for solution in solutions:
        filename = filename_builder(
            solution, impl, dataset, recall_min, qps_min
        )
        results = load_search_results(solution, filename, seed=MOCK_SEED)
        results_combi[solution] = results

    #* 2. Determine metric type and value for file naming and plotting
    metric_type = "recall" if recall_min is not None else "qps"
    metric_value = recall_min if recall_min is not None else qps_min

    #* 3. Plotting accumulated_timestamp
    plot_multi_accumulated_timestamp(
        results=results_combi,
        dirname="all",
        filename=f"{impl}_{dataset}_{metric_type}_{metric_value}_accumulated.png",
        recall_min=recall_min,
        qps_min=qps_min,
        tuning_budget=TUNING_BUDGET,
    )

    #* 4. Save Optimal Hyperparameters of each solution
    for solution, results in results_combi.items():
        optimal_combi[solution] = get_optimal_hyperparameter(
            results, recall_min=recall_min, qps_min=qps_min
        )
    
    save_optimal_hyperparameters(
        optimal_combi,
        recall_min=recall_min,
        qps_min=qps_min,
        seed=MOCK_SEED,
    )
    
    # ! 5) TODO for the combined logic can be placed here


def main():
    SOLUTIONS = [
        "grid_search",
        "random_search",
        "our_solution",
    ]
    IMPLS = [
        "faiss",
    ]
    DATASETS = [
        "nytimes-256-angular",
    ]
    RECALL_MINS = [0.95]
    QPS_MINS = [10000]

    # Create a single combined iterator for all jobs
    all_iters = itertools.product(IMPLS, DATASETS)

    for impl, dataset in all_iters:
        print(f"--- Processing {impl} for {dataset} ---")

        # Process for each recall_min value
        for recall_min in RECALL_MINS:
            print(f"  - Metric: recall_min = {recall_min}")
            _process_single_metric(
                impl=impl, 
                dataset=dataset, 
                solutions=SOLUTIONS, 
                recall_min=recall_min
            )

        # Process for each qps_min value
        for qps_min in QPS_MINS:
            print(f"  - Metric: qps_min = {qps_min}")
            _process_single_metric(
                impl=impl,
                dataset=dataset,
                solutions=SOLUTIONS,
                qps_min=qps_min
            )
                
if __name__ == "__main__":
    main()