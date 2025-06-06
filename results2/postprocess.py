from main.constants import TUNING_BUDGET
from main.utils import filename_builder, load_search_results, \
    plot_multi_accumulated_timestamp, plot_searched_points_3d, plot_timestamp

def process_file():
    SOLUTION = "vd_tuner"
    FILENAME = "vd_tuner_hnswlib_nytimes-256-angular_8h_False_3.csv"
    TUNING_BUDGET = 3600 * 8
    RECALL_MIN = 0.95
    results = load_search_results(SOLUTION, FILENAME)
    _FILENAME = FILENAME.split(".csv")[0]
    plot_timestamp(results, SOLUTION, f"{_FILENAME}_timestamp_plot.png", recall_min=RECALL_MIN)
    plot_searched_points_3d(results, SOLUTION, f"{_FILENAME}_searched_points_3d.png", recall_min=RECALL_MIN)

def process_combi():
    SOLUTIONS = [
        "grid_search",
        # "grid_search_heuristic",
        "random_search",
        # "random_search_heuristic",
        "our_solution7",
        # "vd_tuner",
        # "our_solution4",
        # "our_solution6",
    ]
    IMPLS = [
        "hnswlib",
        "faiss",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "youtube-1024-angular",
        "msmarco-384-angular",
        "dbpediaentity-768-angular",
    ]
    RECALL_MINS = [
        0.90,
        0.95,
        0.975,
    ]
    for impl in IMPLS:
        for dataset in DATASETS:
            for recall_min in RECALL_MINS:
                results_combi = dict()
                for solution in SOLUTIONS:
                    filename = filename_builder(
                        solution, impl, dataset, recall_min
                    )
                    results = load_search_results(solution, filename)
                    results_combi[solution] = results
                plot_multi_accumulated_timestamp(
                    results=results_combi, 
                    dirname="naive",
                    filename=f"{impl}_{dataset}_{recall_min}_accumulated.png", 
                    recall_min=recall_min,
                    tuning_budget=TUNING_BUDGET,
                )
                
if __name__ == "__main__":
    # process_file()
    process_combi()