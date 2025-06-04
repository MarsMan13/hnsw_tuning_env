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
    
def get_optimal_hyperparameter(results, recall_min=0.95):
    optimal_hyperparameters = None
    max_qps = 0.0
    for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
        if recall >= (recall_min - TOLERANCE - 1e-4) and qps > max_qps:
            max_qps = qps
            if type(tuning_time) is not float:
                optimal_hyperparameters = ((M, efC, efS), 
                                            (round(tuning_time.item(),2), round(recall.item(), 3), round(qps.item(), 2),
                                            round(total_time.item(), 2), round(build_time.item(), 2), int(index_size)))
            else:
                optimal_hyperparameters = ((M, efC, efS), 
                                            (round(tuning_time, 2), round(recall, 3), round(qps, 2),
                                            round(total_time, 2), round(build_time, 2), int(index_size)))
    if optimal_hyperparameters is None:
        print("No hyperparameters found with recall >= {recall_min}")
    return optimal_hyperparameters

def get_local_optimal_hyperparameter(results, recall_min=0.95):
    Ms = list(set([M for (M, efC, efS), _ in results])); Ms.sort()
    get_local_optimal_hyperparameters = []
    for cur_M in Ms:
        max_qps = 0.0
        local_opt = None
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if M == cur_M and qps > max_qps and recall >= recall_min:
                max_qps = qps
                if type(tuning_time) is not float:
                    local_opt = ((M, efC, efS), 
                                (round(tuning_time.item(), 2), round(recall.item(), 3), round(qps.item(), 2),
                                round(total_time.item(), 2), round(build_time.item(), 2), int(index_size)))
                else:
                    local_opt = ((M, efC, efS), 
                                (round(tuning_time, 2), round(recall, 3), round(qps, 2),
                                round(total_time, 2), round(build_time, 2), int(index_size)))
        if local_opt : get_local_optimal_hyperparameters.append(local_opt)
    if not get_local_optimal_hyperparameters:
        print(f"No hyperparameters found with recall >= {recall_min}")
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

def plot_timestamp(results, solution, filename, recall_min, tuning_budget, accumulated=False, ylabel='QPS'):
    save_path = _save_path("timestamp" if not accumulated else "acc_timestamp", solution, filename)
    filtered_results = [
        (value[0], value[2])  # T_record, qps
        for _, value in results
        if value[1] >= recall_min and value[0] <= tuning_budget
    ]
    # filtered_results.append((tuning_budget, 0))  # Add a point at the end of the tuning budget
    if accumulated:
        max_qps = 0.0
        filtered_results = []
        for i in range(len(results)):
            M, efC, efS = results[i][0]
            T_record, recall, qps, total_time, build_time, index_size = results[i][1]
            if qps > max_qps and recall >= recall_min and T_record <= tuning_budget:
                print(f"M: {M:3}, efC: {efC:3}, efS: {efS:4} || T_record: {T_record}, recall: {recall}, qps: {qps}")
                max_qps = qps
                filtered_results.append((T_record, qps))
        filtered_results.append((tuning_budget, max_qps))  # Add a point at the end of the tuning budget
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

def plot_searched_points_3d(results, solution, filename, recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET):
    save_path = _save_path("searched_points", solution, filename)

    M_vals = []
    efC_vals = []
    qps_vals = []
    filtered_results = [
        (*key, value[0], value[2])  # T_record, qps
        for key, value in results
        if value[1] >= recall_min and value[0] <= tuning_budget
    ]
    for M, efC, efS, recall, qps in filtered_results:
        M_vals.append(M)
        efC_vals.append(efC)
        qps_vals.append(qps)

    M_vals = np.array(M_vals)
    efC_vals = np.array(efC_vals)
    qps_vals = np.array(qps_vals)

    fig = plt.figure(figsize=(20, 20))
    views = [(30, -120), (30, -210), (30, -300), (30, 30), (45, 180), (45, 0), (30, 270), (225, 60), (270, 90)]

    global_opt = get_optimal_hyperparameter(results, recall_min=recall_min)
    local_opts = get_local_optimal_hyperparameter(results, recall_min=recall_min)

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        ax.plot_trisurf(efC_vals, M_vals, qps_vals, cmap=cm.viridis, alpha=0.95, edgecolor='none')

        # Plot all local optimal points in red
        for hp, perf in local_opts:
            M, efC, efS = hp
            if (M, efC) == global_opt[0][:2]:
                continue
            _, _, qps, *_ = perf
            ax.scatter(efC, M, qps+50, color='red', s=50)
        ax.scatter(global_opt[0][1], global_opt[0][0], global_opt[1][2]+250, color='blue', s=250)
        ax.set_title(f'View {i + 1}')
        ax.set_xlabel('efC')
        ax.set_ylabel('M')
        ax.set_zlabel('QPS')
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_efS_3d(results, solution, filename, recall_min=0.95, tuning_budget=TUNING_BUDGET):
    save_path = _save_path("efS_3d", solution, filename)
    
    M_vals = []
    efC_vals = []
    efS_vals = []
    filtered_results = [
        (*key, value[0], value[2])  # T_record, qps
        for key, value in results
        if value[1] >= recall_min and value[0] <= tuning_budget
    ]
    for M, efC, efS, recall, qps in filtered_results:
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
