import os
import csv
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from src.constants import EFC_MAX, EFC_MIN, M_MAX, M_MIN, RECALL_MIN, SEED, TOLERANCE, TUNING_BUDGET, MAX_SAMPLING_COUNT

def filename_builder(solution, impl, dataset, recall_min, qps_min, tuning_budget=None):
    if tuning_budget:
        return f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q_{tuning_budget}.csv"
    else:
        return f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.csv"

def _save_path(output_type, solution, filename, seed, sampling_count):
    path = os.path.join("results", output_type, str(seed), str(sampling_count), solution)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

def is_already_saved(solution, filename, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    save_path = _save_path("result", solution, filename, seed, sampling_count)
    return os.path.exists(save_path)

def save_search_results(results, solution, filename, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    save_path = _save_path("result", solution, filename, seed, sampling_count)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "efC", "efS", "T_record", "recall", "qps", "total_time", "build_time", "index_size"])
        for (M, efC, efS), (T_record, recall, qps, total_time, build_time, index_size) in results:
            writer.writerow([M, efC, efS, T_record, recall, qps, total_time, build_time, index_size])

def load_search_results(solution, filename, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    load_path = _save_path("result", solution, filename, seed, sampling_count)
    results = []
    if not os.path.exists(load_path):
        print(f"File {load_path} does not exist.")
        return results
    with open(load_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            M, efC, efS = map(int, row[:3])
            T_record, recall, qps, total_time, build_time = map(float, row[3:8])
            index_size = int(float(row[8]))
            results.append(((M, efC, efS), (T_record, recall, qps, total_time, build_time, index_size)))
    return results
    
def get_optimal_hyperparameter(results, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    
    optimal_hyperparameters = ((None, None, None), (0.0, 0.0, 0.0, 0.0, 0.0, 0))

    if recall_min is not None:
        best_qps = 0.0
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if recall >= recall_min and qps > best_qps:
                best_qps = qps
                optimal_hyperparameters = ((M, efC, efS), (
                    round(float(tuning_time), 2), round(float(recall), 3), round(float(qps), 2),
                    round(float(total_time), 2), round(float(build_time), 2), int(index_size)
                ))

    elif qps_min is not None:
        best_recall = 0.0
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if qps >= qps_min and recall > best_recall:
                best_recall = recall
                optimal_hyperparameters = ((M, efC, efS), (
                    round(float(tuning_time), 2), round(float(recall), 3), round(float(qps), 2),
                    round(float(total_time), 2), round(float(build_time), 2), int(index_size)
                ))

    return optimal_hyperparameters

def get_local_optimal_hyperparameter(results, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."

    Ms = list(set([M for (M, efC, efS), _ in results]))
    Ms.sort()
    get_local_optimal_hyperparameters = []

    for cur_M in Ms:
        best_metric = 0.0
        local_opt = None
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if M != cur_M:
                continue
            if recall_min is not None:
                if recall >= (recall_min - TOLERANCE - 1e-5) and qps > best_metric:
                    best_metric = qps
                else:
                    continue
            elif qps_min is not None:
                if qps >= (qps_min - TOLERANCE - 1e-5) and recall > best_metric:
                    best_metric = recall
                else:
                    continue
            if hasattr(recall, "item"):
                local_opt = (
                    (M, efC, efS),
                    (
                        round(tuning_time.item(), 2),
                        round(recall.item(), 3),
                        round(qps.item(), 2),
                        round(total_time.item(), 2),
                        round(build_time.item(), 2),
                        int(index_size),
                    ),
                )
            else:
                local_opt = (
                    (M, efC, efS),
                    (
                        round(tuning_time, 2),
                        round(recall, 3),
                        round(qps, 2),
                        round(total_time, 2),
                        round(build_time, 2),
                        int(index_size),
                    ),
                )
        if local_opt:
            get_local_optimal_hyperparameters.append(local_opt)
    if not get_local_optimal_hyperparameters:
        print(f"No hyperparameters found with {'recall >= ' + str(recall_min) if recall_min is not None else 'qps >= ' + str(qps_min)}")
    return get_local_optimal_hyperparameters

def save_optimal_hyperparameters(optimal_combi, recall_min=None, qps_min=None, seed=SEED, sampling_count = MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("optimal_hyperparameters", "_".join(optimal_combi.keys()), f"optimal_hyperparameters_{recall_min}r_{qps_min}q.csv", seed, sampling_count)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "M", "efC", "efS", "T_record", "recall", "qps", "total_time", "build_time", "index_size"])
        for item in optimal_combi.items():
            print(item)
        for solution, (hp, perf) in optimal_combi.items():
            M, efC, efS = hp
            T_record, recall, qps, total_time, build_time, index_size = perf
            writer.writerow([solution, M, efC, efS, T_record, recall, qps, total_time, build_time, index_size])

def plot_multi_accumulated_timestamp(results, dirname, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, seed=SEED, sampling_count = MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."

    save_path = _save_path("accumulated_timestamp", dirname, filename, seed, sampling_count)
    
    markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'X', '+', 'v']
    colors = [cm.get_cmap('tab10')(i) for i in range(10)] # tab10 provides 10 distinct colors

    marker_idx = 0
    color_idx = 0

    plt.figure(figsize=(10, 6)) # Set figure size for better readability

    for solution, result in results.items():
        if recall_min:
            filtered = [
                (value[0], value[2])  # (T_record, qps)
                for _, value in result
                if value[1] >= recall_min
            ]
        elif qps_min:
            filtered = [
                (value[0], value[1])  # (T_record, recall)
                for _, value in result
                if value[2] >= qps_min
            ]
        if not filtered:
            if recall_min : print(f"[{solution}] No results with recall >= {recall_min}")
            if qps_min : print(f"[{solution}] No results with qps >= {qps_min}")
            continue
        filtered.insert(0, (0.0, 0.0))
        filtered.sort(key=lambda x: x[0])
        accumulated = [(0.0, 0.0)] # Start with (0,0) for the plot
        max_perf = 0.0
        for t, perf in filtered:
            if perf > max_perf:
                accumulated.append((t, max_perf))
                max_perf = perf
                accumulated.append((t, perf))
        if not accumulated:
            continue
        accumulated.append((tuning_budget, max_perf))
        timestamps, perf_values = zip(*accumulated)
        plt.plot(
            timestamps,
            perf_values,
            marker=markers[marker_idx % len(markers)],  # Cycle through markers
            color=colors[color_idx % len(colors)],      # Cycle through colors
            label=solution
        )
        marker_idx += 1
        color_idx += 1
    plt.title("Accumulated TimeStamp Plot")
    plt.xlabel("Time (seconds)")
    plt.ylabel("QPS" if recall_min else "Recall")
    plt.xlim(0, tuning_budget)
    plt.grid(True) # Add grid for better readability
    plt.legend(loc='best') # Display legend, choosing the best location automatically
    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    plt.savefig(save_path)
    plt.close()

def plot_timestamp(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("timestamp", solution, filename, seed, sampling_count)
    if recall_min:
        filtered_results = [
            (value[0], value[2])  # T_record, qps
            for _, value in results
            if value[1] >= recall_min and value[0] <= tuning_budget
        ]
    if qps_min:
        filtered_results = [
            (value[0], value[1])
            for _, value in results
            if value[2] >= qps_min and value[0] <= tuning_budget
        ]
    # filtered_results.append((tuning_budget, 0))  # Add a point at the end of the tuning budget
    if not filtered_results:
        print(f"No results with recall >= {recall_min}")
        return
    filtered_results.append((tuning_budget, filtered_results[-1][1]))  # Add a point at the end of the tuning budget
    #! TODO : Debug it
    # filtered_results = filtered_results[:-1]
    # print(filtered_results)
    timestamps, qps_values = zip(*filtered_results)
    plt.plot(timestamps, qps_values, marker='o', label=solution)
    plt.title("TimeStamp Plot")
    plt.xlabel("Time")
    plt.ylabel("QPS" if recall_min else "Recall")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_searched_points_3d(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("searched_points", solution, filename, seed, sampling_count)
    M_vals = []
    efC_vals = []
    perf_vals = []
    if recall_min:
        filtered_results = [
            (hp[0], hp[1], perf[2])  # T_record, qps
            for hp, perf in results
            if perf[0] <= tuning_budget
            # if perf[1] >= recall_min and perf[0] <= tuning_budget
        ]
    if qps_min:
        filtered_results = [
            (hp[0], hp[1], perf[1])
            for hp, perf in results
            if perf[0] <= tuning_budget
            # if perf[2] >= qps_min and perf[0] <= tuning_budget
        ]
    for M, efC, perf in filtered_results:
        M_vals.append(M)
        efC_vals.append(efC)
        perf_vals.append(perf)

    M_vals = np.array(M_vals)
    efC_vals = np.array(efC_vals)
    perf_vals = np.array(perf_vals)

    fig = plt.figure(figsize=(20, 20))
    views = [(30, -120), (30, -210), (30, -300), (30, 30), (45, 180), (45, 0), (30, 270), (225, 60), (270, 90)]
    
    global_opt = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
    local_opts = get_local_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        #! TODO: We Use It?
        # ax.plot_trisurf(efC_vals, M_vals, perf_vals, cmap=cm.viridis, alpha=0.95, edgecolor='none')
        ax.scatter(efC_vals, M_vals, perf_vals, c=perf_vals, cmap=cm.viridis, s=20, alpha=0.7)
        for x, y, z in zip(efC_vals, M_vals, perf_vals):
            ax.plot([x, x], [y, y], [0, z], color='gray', alpha=0.3, linewidth=1)  # Vertical lines to the ground plane
            if z > 0:   ax.scatter(x, y, 0, color='gray', alpha=0.3, s=5)  # Scatter points on the ground plane
            else:       ax.scatter(x, y, 0, color="black", alpha=0.95, s=25)
        # Plot all local optimal points in red
        for hp, perf in local_opts:
            M, efC, efS = hp
            if (M, efC) == global_opt[0][:2]:
                continue
            _, recall, qps, *_ = perf
            ax.scatter(efC, M, qps if recall_min else recall, color='red', s=50)
        ax.scatter(global_opt[0][1], global_opt[0][0], global_opt[1][2] if recall_min else global_opt[1][1], color='blue', s=250)
        ax.set_title(f'View {i + 1}')
        ax.set_xlabel('efC')
        ax.set_ylabel('M')
        ax.set_zlabel('QPS' if recall_min else 'Recall')
        ax.set_xlim(EFC_MIN, EFC_MAX)
        ax.set_ylim(M_MIN, M_MAX)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(save_path)

def plot_efS_3d(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("efS_3d", solution, filename, seed, sampling_count)
    
    M_vals = []
    efC_vals = []
    efS_vals = []
    if recall_min:
        filtered_results = [
            hp
            for hp, perf  in results
            if perf[0] <= tuning_budget
        ]
    if qps_min:
        filtered_results = [
            hp
            for hp, perf in results
            if perf[0] <= tuning_budget
        ]
    for M, efC, efS in filtered_results:
        M_vals.append(M)
        efC_vals.append(efC)
        efS_vals.append(efS)

    M_vals = np.array(M_vals)
    efC_vals = np.array(efC_vals)
    efS_vals = np.array(efS_vals)

    fig = plt.figure(figsize=(20, 20))
    views = [(30, -120), (30, -210), (30, -300), (30, 30), (45, 180), (45, 0), (30, 270), (225, 60), (270, 90)]

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        # ax.plot_trisurf(efC_vals, M_vals, efS_vals, cmap=cm.viridis, alpha=0.95, edgecolor='none')
        ax.scatter(efC_vals, M_vals, efS_vals, c=efS_vals, cmap=cm.viridis, s=20, alpha=0.7)
        for x, y, z in zip(efC_vals, M_vals, efS_vals):
            ax.plot([x, x], [y, y], [0, z], color='gray', alpha=0.3, linewidth=1)  # Vertical lines to the ground plane
            if z > 0:   ax.scatter(x, y, 0, color='gray', alpha=0.3, s=5)  # Scatter points on the ground plane
            else:       ax.scatter(x, y, 0, color="black", alpha=0.95, s=25)
        ax.set_title(f'View {i+1}')
        ax.set_xlabel('efC')
        ax.set_ylabel('M')
        ax.set_zlabel('efS')
        ax.set_xlim(EFC_MIN, EFC_MAX)
        ax.set_ylim(M_MIN, M_MAX)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
