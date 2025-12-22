import matplotlib.pyplot as plt

from src.constants import TUNING_BUDGET
from src.utils import filename_builder, get_optimal_hyperparameter, \
    load_search_results, plot_accumulated_timestamp_on_ax
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"

MOCK_SEED = 0
def get_results(
    impl: str,
    dataset: str,
    solutions: list,
    recall_min: float = None,
    qps_min: int = None,
    sampling_count: int = None,
    tuning_time: int = TUNING_BUDGET,
):

    results_combi = {}
    #* 1. Load results for all solutions under the given condition
    for solution in solutions:
        filename = filename_builder(
            solution, impl, dataset, recall_min, qps_min
        )
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)
        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(
                results, recall_min=recall_min, qps_min=qps_min
            )
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            optimal_hp = (hp, perf)
            results = [optimal_hp]  # For brute_force, we only keep the optimal hyperparameter
        else:
            results = [result for result in results if result[1][0] <= tuning_time]  # Filter results by tuning time
        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results":results_combi
    }

# FILE: main.py

def main():
    import matplotlib.font_manager as fm
    # 1. Path to your .ttf font file.
    #    Make sure this path is correct.
    font_path = f'{current_dir}/LinLibertine_R.ttf'

    # 2. Register font if it's not already registered.
    if font_path not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path)
    
    # 3. Set the registered font as the default.
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name

    # 4. Ensure the minus sign is displayed correctly in plots.
    plt.rcParams['axes.unicode_minus'] = False
    SOLUTIONS = [
        "brute_force", "our_solution", "grid_search",
        "random_search", "vd_tuner", "optuna", "nsga"
    ]
    # NOTE: To match the requested labels (hnswlib top, faiss bottom),
    #       we arrange the IMPLS list in that order.
    IMPLS = [
        "milvus"
    ]
    DATASETS = [
        "nytimes-256-angular", "glove-100-angular", "sift-128-euclidean",
    ]
    # NOTE: Labels for each column as requested by the user.
    COLUMN_LABELS = [
        "nytimes", "glove", "sift"
    ]
    SAMPLING_COUNT = [10]
    RECALL_MINS = [0.95]

    # --- Task generation logic (same as before) ---
    tasks = []
    for impl in IMPLS:
        for metric in ["recall_min", "qps_min"]:
            for dataset in DATASETS:
                for sampling_count in SAMPLING_COUNT:
                    if metric == "recall_min":
                        for recall_min in RECALL_MINS:
                            task_args = (impl, dataset, SOLUTIONS, recall_min, None, sampling_count)
                            tasks.append(task_args)
                    else:
                        qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)["q75"]
                        task_args = (impl, dataset, SOLUTIONS, None, qps_min, sampling_count)
                        tasks.append(task_args)
    results = []
    for impl, dataset, solutions, recall_min, qps_min, sampling_count in tasks:
        results.append(get_results(
            impl=impl, dataset=dataset, solutions=solutions,
            recall_min=recall_min, qps_min=qps_min, sampling_count=sampling_count
        ))
    
    # --- Plotting logic with requested modifications ---

    # Create a 4x5 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))

    # Plot data on each subplot
    for ax, result in zip(axes.flat, results):
        plot_accumulated_timestamp_on_ax(
            ax, result["results"], result["recall_min"], result["qps_min"]
        )

    # 1. Gather unique handles and labels for the main legend
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))

    # 2. Create the main legend at the top
    fig.legend(
        by_label.values(), by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=5, fontsize=18, frameon=False
    )

    # 3. Add vertical row labels on the left side of the figure
    fig.text(0.02, 0.70, 'hnswlib', va='center', ha='center', rotation='vertical', fontsize=24)
    fig.text(0.02, 0.30, 'faiss',   va='center', ha='center', rotation='vertical', fontsize=24)

    # 4. Add column labels at the bottom of the figure
    for i, label in enumerate(COLUMN_LABELS):
        axes[1, i].set_xlabel(label, fontsize=20, labelpad=10)

    # 5. Adjust subplot layout to prevent overlap and make space for new labels
    plt.subplots_adjust(
        left=0.07,   # Make space for vertical labels
        bottom=0.1,  # Make space for column labels
        top=0.93,    # Make space for the top legend
        wspace=0.3,
        hspace=0.4   # Adjust vertical space between plots
    )

    fig.savefig("milvus_figure.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly,
    # especially on Windows and macOS. It prevents child processes from
    # re-importing and re-executing the main script's code.
    main()