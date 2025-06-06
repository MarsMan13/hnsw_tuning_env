import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

from main.constants import RECALL_MIN, TOLERANCE, TUNING_BUDGET

def filename_builder(solution, impl, dataset, recall_min, tuning_budget=None):
    if tuning_budget:
        return f"{solution}_{impl}_{dataset}_{recall_min}r_{tuning_budget}.csv"
    else:
        return f"{solution}_{impl}_{dataset}_{recall_min}r.csv"

def _save_path(type_, solution, filename):
    path = os.path.join("results", type_, solution)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

def save_search_results(results, solution, filename):
    save_path = _save_path("result", solution, filename)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "efC", "efS", "T_record", "recall", "qps", "total_time", "build_time", "index_size"])
        for (M, efC, efS), (T_record, recall, qps, total_time, build_time, index_size) in results:
            writer.writerow([M, efC, efS, T_record, recall, qps, total_time, build_time, index_size])

def load_search_results(solution, filename):
    load_path = _save_path("result", solution, filename)
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
    
    optimal_hyperparameters = None

    if recall_min is not None:
        best_qps = 0.0
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if recall >= (recall_min - TOLERANCE - 1e-5) and qps > best_qps:
                best_qps = qps
                optimal_hyperparameters = ((M, efC, efS), (
                    round(float(tuning_time), 2), round(float(recall), 3), round(float(qps), 2),
                    round(float(total_time), 2), round(float(build_time), 2), int(index_size)
                ))

    elif qps_min is not None:
        best_recall = 0.0
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if qps >= (qps_min - TOLERANCE - 1e-5) and recall > best_recall:
                best_recall = recall
                optimal_hyperparameters = ((M, efC, efS), (
                    round(float(tuning_time), 2), round(float(recall), 3), round(float(qps), 2),
                    round(float(total_time), 2), round(float(build_time), 2), int(index_size)
                ))

    if optimal_hyperparameters is None:
        print(f"No hyperparameters found with recall >= {recall_min}" if recall_min is not None else f"No hyperparameters found with qps >= {qps_min}")
    
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
            if type(tuning_time) is not float:
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

def plot_multi_accumulated_timestamp(results, dirname, filename, recall_min, tuning_budget, ylabel='QPS'):
    save_path = _save_path("accumulated_timestamp", dirname, filename)
    
    for solution, result in results.items():
        filtered = [
            (value[0], value[2])  # (T_record, qps)
            for _, value in result
            if value[1] >= recall_min
        ]
        if not filtered:
            print(f"[{solution}] No results with recall >= {recall_min}")
            continue

        filtered.sort(key=lambda x: x[0])

        accumulated = [(0.0, 0.0)]
        max_qps = 0.0
        for t, qps in filtered:
            if qps > max_qps:
                accumulated.append((t, max_qps))
                max_qps = qps
                accumulated.append((t, qps))
        if not accumulated:
            continue
        accumulated.append((tuning_budget, max_qps))  # Add a point at the end of the tuning budget
        timestamps, qps_values = zip(*accumulated)
        plt.plot(timestamps, qps_values, marker='o', label=solution)

    plt.title("Accumulated TimeStamp Plot")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_timestamp(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, ylabel='QPS'):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("timestamp", solution, filename)
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
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_searched_points_3d(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("searched_points", solution, filename)
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
        ax.plot_trisurf(efC_vals, M_vals, perf_vals, cmap=cm.viridis, alpha=0.95, edgecolor='none')
        # Plot all local optimal points in red
        for hp, perf in local_opts:
            M, efC, efS = hp
            if (M, efC) == global_opt[0][:2]:
                continue
            _, recall, qps, *_ = perf
        #     ax.scatter(efC, M, qps if recall_min else recall, color='red', s=50)
        # ax.scatter(global_opt[0][1], global_opt[0][0], global_opt[1][2] if recall_min else global_opt[1][1], color='blue', s=250)
        ax.set_title(f'View {i + 1}')
        ax.set_xlabel('efC')
        ax.set_ylabel('M')
        ax.set_zlabel('QPS' if recall_min is None else 'Recall')
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(save_path)

def plot_efS_3d(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("efS_3d", solution, filename)
    
    M_vals = []
    efC_vals = []
    efS_vals = []
    if recall_min:
        filtered_results = [
            hp
            for hp, perf  in results
            if perf[1] >= recall_min and perf[0] <= tuning_budget
        ]
    if qps_min:
        filtered_results = [
            hp
            for hp, perf in results
            if perf[2] >= qps_min and perf[0] <= tuning_budget
        ]
    for M, efC, efS in filtered_results:
        M_vals.append(M)
        efC_vals.append(efC)
        efS_vals.append(efS)

    M_vals = np.array(M_vals)
    efC_vals = np.array(efC_vals)
    efS_vals = np.array(efS_vals)

    fig = plt.figure(figsize=(12, 10))
    views = [(30, -120), (30, -210), (30, -300), (30, 30)]

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_trisurf(efC_vals, M_vals, efS_vals, cmap=cm.viridis, alpha=0.95, edgecolor='none')
        ax.set_title(f'View {i+1}')
        ax.set_xlabel('efC')
        ax.set_ylabel('M')
        ax.set_zlabel('efS')
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
