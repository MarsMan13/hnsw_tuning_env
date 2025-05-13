from main.utils import load_search_results, plot_searched_points_3d, plot_timestamp

if __name__ == "__main__":
    SOLUTION = "vd_tuner"
    FILENAME = "vd_tuner_hnswlib_nytimes-256-angular_8h_False_3.csv"
    TUNING_BUDGET = 3600 * 8
    MIN_RECALL = 0.95
    results = load_search_results(SOLUTION, FILENAME)
    _FILENAME = FILENAME.split(".csv")[0]
    plot_timestamp(results, SOLUTION, f"{_FILENAME}_timestamp_plot.png", min_recall=MIN_RECALL)
    plot_timestamp(results, SOLUTION, f"{_FILENAME}_timestamp_plot_accumulated.png", min_recall=MIN_RECALL, accumulated=True)
    plot_searched_points_3d(results, SOLUTION, f"{_FILENAME}_searched_points_3d.png", min_recall=MIN_RECALL)
