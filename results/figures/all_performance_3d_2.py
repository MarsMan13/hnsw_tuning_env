import itertools
import multiprocessing
import os
import matplotlib

# 우분투 서버(GUI 없는 환경) 필수 설정
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- 기존 src/utils.py 및 constants에서 가져오는 모듈 가정 ---
from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder, 
    load_search_results,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset
from src.constants import (
    M_MIN, M_MAX,
    EFC_MIN, EFC_MAX,
    EFS_MIN, EFS_MAX,
)
from data.ground_truths.ground_truth import GroundTruth

# 전역 상수 설정
MOCK_SEED = "42"

# ==========================================
# [New Function] 2D Optimal Curve Plotter
# ==========================================
def _plot_and_save_optimal_curve(M_efc_to_perf, impl, dataset, metric_type, metric_val, seed, pid):
    """
    Groups data by M, finds the efc with maximum performance for each M,
    and plots the (M, optimal_efc) curve in 2D.
    """
    # 1. Find optimal efc for each M
    # Dictionary structure: M -> (max_perf, best_efc)
    m_best_map = {}
    
    for (m, efc), perf in M_efc_to_perf.items():
        if m not in m_best_map:
            m_best_map[m] = (perf, efc)
        else:
            current_max_perf, _ = m_best_map[m]
            if perf > current_max_perf:
                m_best_map[m] = (perf, efc)
            elif perf == current_max_perf:
                # Tie-breaking: if performance is same, choose smaller efc (efficiency)
                # This is optional, but reasonable for ANNS
                if efc < m_best_map[m][1]:
                    m_best_map[m] = (perf, efc)

    # 2. Sort by M for plotting
    sorted_ms = sorted(m_best_map.keys())
    optimal_efcs = [m_best_map[m][1] for m in sorted_ms]
    
    if not sorted_ms:
        print(f"[PID {pid}] No data to plot 2D curve.")
        return

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points and connect them
    ax.plot(sorted_ms, optimal_efcs, marker='o', linestyle='-', color='b', label='Optimal efc', linewidth=2, markersize=8)
    
    # Annotate points (Optional: show efc value near the point)
    for m, efc in zip(sorted_ms, optimal_efcs):
        ax.annotate(f'{efc}', (m, efc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    ax.set_xlabel('M parameter', fontsize=12)
    ax.set_ylabel('Optimal efConstruction (yielding max perf)', fontsize=12)
    ax.set_title(f'Optimal Hyperparameter Curve (M vs efc)\n{impl} / {dataset} ({metric_type} >= {metric_val})', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()

    # 4. Save
    # Separate directory for 2D plots
    output_dir = os.path.join("results", "plots_2d_optimal", str(seed))
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{impl}_{dataset}_{metric_type}_{metric_val}_optimal_curve.png"
    save_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[PID {pid}] Saved 2D Curve: {output_filename}")


def _plot_and_save_surface(M_efc_to_perf, impl, dataset, metric_type, metric_val, seed, pid):
    """
    3D Surface 그래프를 그리고 파일로 저장하는 헬퍼 함수
    """
    # 리스트를 numpy array로 변환 (trisurf 사용을 위해 권장됨)
    X_m = []
    Y_efc = []
    Z_perf = []
    
    for key, perf in M_efc_to_perf.items():
        m_val, efc_val = key
        X_m.append(m_val)
        Y_efc.append(efc_val)
        Z_perf.append(perf)
        
    X_m = np.array(X_m)
    Y_efc = np.array(Y_efc)
    Z_perf = np.array(Z_perf)

    # 2x2 subplot 설정
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'3D Performance Surface: {impl} / {dataset}\n(Metric: {metric_type} >= {metric_val})', fontsize=20)

    # 4방향 뷰 포인트 (Elevation, Azimuth)
    view_angles = [
        (30, 45),   # Front-Right
        (30, 135),  # Back-Right
        (30, 225),  # Back-Left
        (30, 315)   # Front-Left
    ]
    view_names = ["Front-Right", "Back-Right", "Back-Left", "Front-Left"]

    # 그래프 그리기
    surf = None
    for idx, (elev, azim) in enumerate(view_angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        # Scatter 대신 plot_trisurf 사용 (시커먼 선 제거)
        surf = ax.plot_trisurf(X_m, Y_efc, Z_perf, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
        
        ax.set_xlabel('M parameter', labelpad=10)
        ax.set_ylabel('efConstruction parameter', labelpad=10)
        ax.set_zlabel('Performance (QPS/Recall)', labelpad=10)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'{view_names[idx]} (Azim={azim}°)', fontsize=14)

    # 공통 Colorbar
    if surf:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(surf, cax=cbar_ax, label='Performance Value')

    # 저장 경로 설정
    output_dir = os.path.join("results", "plots_3d_surface", str(seed))
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{impl}_{dataset}_{metric_type}_{metric_val}_3d_surface.png"
    save_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig) 
    
    print(f"[PID {pid}] Saved 3D Surface: {output_filename}")


def _3d_performance_plotter(impl, dataset, recall_min, qps_min, sampling_count):
    """
    개별 프로세스에서 실행될 작업 함수입니다.
    데이터를 로드하고 처리한 후 Plotting 함수들을 호출합니다.
    """
    try:
        # 프로세스 ID 출력 (디버깅용)
        pid = multiprocessing.current_process().pid
        metric_type = "Recall" if recall_min is not None else "QPS"
        metric_val = recall_min if recall_min is not None else qps_min
        print(f"[PID {pid}] Processing: {impl}, {dataset}, {metric_type}={metric_val}")

        assert (recall_min is None) != (qps_min is None), "Either recall_min or qps_min must be specified."
        
        gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=int(sampling_count))
        results = []
        for m in range(M_MIN, M_MAX + 1, 2):
            for efc in range(EFC_MIN, EFC_MAX + 1, 4):
                efs = gd.get_efS(M=m, efC=efc, target_recall=recall_min, target_qps=qps_min)
                recall, qps, total_time, build_time, index_size = gd.get(M=m, efC=efc, efS=efs)
                results.append(((m, efc, efs), (recall, qps)))

        M_efc_to_perf = dict()
        def update_perf(M, efc, perf):
            if (M, efc) not in M_efc_to_perf:
                M_efc_to_perf[(M, efc)] = perf
            else:
                M_efc_to_perf[(M, efc)] = max(M_efc_to_perf[(M, efc)], perf)
                
        for res in results:
            M, efc, efs = res[0]
            target_perf = res[1][0] if qps_min is not None else res[1][1] 
            
            is_feasible = False
            if qps_min is not None and res[1][1] >= qps_min: 
                is_feasible = True
            if recall_min is not None and res[1][0] >= recall_min: 
                is_feasible = True
            
            if is_feasible:
                update_perf(M, efc, target_perf)

        if not M_efc_to_perf:
            print(f"[PID {pid}] No feasible points for {dataset}. Skipping plot.")
            return

        # ---------------------------------------------------------
        # 1. 3D Surface Plot (Original request)
        # ---------------------------------------------------------
        _plot_and_save_surface(
            M_efc_to_perf, 
            impl, 
            dataset, 
            metric_type, 
            metric_val, 
            seed=MOCK_SEED, 
            pid=pid
        )

        # ---------------------------------------------------------
        # 2. [Added] 2D Optimal Curve Plot (Requested)
        # ---------------------------------------------------------
        _plot_and_save_optimal_curve(
            M_efc_to_perf, 
            impl, 
            dataset, 
            metric_type, 
            metric_val, 
            seed=MOCK_SEED, 
            pid=pid
        )

    except Exception as e:
        print(f"[PID {multiprocessing.current_process().pid}] Error processing {impl}-{dataset}: {e}")

def main():
    IMPLS = [
        "faiss",
        # "hnswlib",
    ]
    DATASETS = [
        "nytimes-256-angular-100p",
        "nytimes-256-angular-50p",
        "nytimes-256-angular-10p",
        "nytimes-256-angular-1p",
        # "nytimes-256-angular",
        # "glove-100-angular",
        # "sift-128-euclidean",
        # "youtube-1024-angular",
        # "deep1M-256-angular",
    ]
    SAMPLING_COUNT = [
        "10",
    ]
    RECALL_MINS = [
        0.90,
        0.925,
        0.95,
    ]
    QPS_MINS = [
        # "q75".
        60000,
        80000,
        100000,
    ]

    # --- Task 생성 ---
    tasks = []
    all_iters = itertools.product(IMPLS, DATASETS, SAMPLING_COUNT)
    
    print("--- Generating Tasks ---")
    for impl, dataset, sampling_count in all_iters:
        # 1. Recall Constraint Tasks
        for recall_min in RECALL_MINS:
            task_args = (impl, dataset, recall_min, None, sampling_count)
            tasks.append(task_args)
            print(f"  [Queue] {impl}, {dataset}, recall_min={recall_min}")
            
        # 2. QPS Constraint Tasks
        try:
            # percent_to_qps = get_qps_metrics_dataset(impl, dataset, ret_dict=True)
            # for qps_min in QPS_MINS:
            #     if qps_min in percent_to_qps:
            #         real_qps_val = percent_to_qps[qps_min]
            #         task_args = (impl, dataset, None, real_qps_val, sampling_count)
            #         tasks.append(task_args)
            #         print(f"  [Queue] {impl}, {dataset}, qps_min={qps_min}({real_qps_val})")
            for qps_min in QPS_MINS:
                task_args = (impl, dataset, None, qps_min, sampling_count)
                tasks.append(task_args)
                print(f"  [Queue] {impl}, {dataset}, qps_min={qps_min}({qps_min})")
        except Exception as e:
            print(f"  [Warning] Could not get QPS metrics for {dataset}: {e}")

    # --- Multiprocessing 실행 ---
    num_cpus = multiprocessing.cpu_count()
    num_processes = max(1, num_cpus - 2) 
    
    print(f"\n--- Starting Parallel Processing with {num_processes} workers ---")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(_3d_performance_plotter, tasks)
        
    print("\n--- All plotting tasks completed ---")

if __name__ == "__main__":
    main()