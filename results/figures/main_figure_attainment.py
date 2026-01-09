import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
import numpy as np

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    _feasible_and_objective_factory,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
MOCK_SEED = 0

# --- [스타일 설정] 솔루션별 색상 및 마커 ---
SOL_STYLES = {
    "our_solution":  {"c": "#d62728", "marker": "o", "ls": "-",  "lw": 2.0, "zorder": 10, "label": "CHAT (Ours)"}, # Red
    "vd_tuner":      {"c": "#9467bd", "marker": "s", "ls": "--", "lw": 1.5, "zorder": 5,  "label": "VDTuner"},     # Purple
    "optuna":        {"c": "#8c564b", "marker": "^", "ls": "-.", "lw": 1.5, "zorder": 4,  "label": "Optuna"},      # Brown
    "nsga":          {"c": "#e377c2", "marker": "D", "ls": ":",  "lw": 1.5, "zorder": 3,  "label": "NSGA-II"},     # Pink
    "random_search": {"c": "#7f7f7f", "marker": "v", "ls": "--", "lw": 1.2, "zorder": 2,  "label": "Random"},      # Gray
    "grid_search":   {"c": "#bcbd22", "marker": "P", "ls": ":",  "lw": 1.2, "zorder": 1,  "label": "Grid"},        # Olive
}
SOL_ORDER = ["our_solution", "vd_tuner", "optuna", "nsga", "random_search", "grid_search"]

ALPHAS = [0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99]

def short_ds(name: str) -> str:
    return name.split("-")[0]

def get_results(impl, dataset, solutions, recall_min=None, qps_min=None, sampling_count=None, tuning_time=TUNING_BUDGET):
    assert (recall_min is not None) != (qps_min is not None)
    results_combi = {}
    oracle_best_metric = 0.0

    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)

        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            results = [(hp, perf)]
            
            # Oracle Best Metric 계산
            metric = qps if recall_min is not None else recall
            if metric > oracle_best_metric:
                oracle_best_metric = metric
        else:
            results = [r for r in results if r[1][0] <= tuning_time]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "oracle_best_metric": oracle_best_metric,
    }

def calculate_attainment_times(data, is_feasible, objective, oracle_best, budget):
    """
    각 Alpha 목표치에 도달한 최초 시간(t_hit)을 계산하여 반환.
    도달하지 못한 경우 None 반환.
    """
    if not data or oracle_best <= 1e-9:
        return [None] * len(ALPHAS)

    # 1. 유효한 (Time, Perf) 추출 및 시간순 정렬
    valid_points = []
    for _, perf in data:
        if is_feasible(perf):
            val = objective(perf)
            t = perf[0]
            valid_points.append((t, val))
    valid_points.sort(key=lambda x: x[0])

    # 2. Best-so-far Curve 생성
    times = []
    best_perfs = []
    current_max = 0.0
    for t, val in valid_points:
        if t > budget:
            break
        if val > current_max:
            current_max = val
            times.append(t)
            best_perfs.append(current_max)

    # 3. 각 Alpha에 대해 t_hit 찾기
    t_hits = []
    for alpha in ALPHAS:
        target_val = alpha * oracle_best
        found_t = None
        
        # Best-so-far 곡선을 순회하며 목표 달성 시점 찾기
        for t, perf in zip(times, best_perfs):
            if perf >= target_val:
                found_t = t
                break
        
        t_hits.append(found_t)
    
    return t_hits

def build_attainment_points(data, is_feasible, objective, oracle_best, budget,
                            alphas_base=ALPHAS_BASE, tail_start=TAIL_START):
    """
    Returns (plot_xs, plot_ys)
    - For alphas_base (<=0.99): first hit times as before (y=alpha)
    - For tail [tail_start, 1.0]: add a point whenever best-so-far improves,
      using y = best/oracle_best (clipped to 1.0)
    """
    if not data or oracle_best <= 1e-12:
        return [], []

    # 1) feasible points sorted by time
    valid_points = []
    for _, perf in data:
        if is_feasible(perf):
            t = float(perf[0])
            if t <= budget:
                val = float(objective(perf))
                valid_points.append((t, val))
    valid_points.sort(key=lambda x: x[0])

    # 2) best-so-far breakpoints
    times = []
    best_perfs = []
    current_max = -np.inf
    for t, val in valid_points:
        if val > current_max:
            current_max = val
            times.append(t)
            best_perfs.append(current_max)

    plot_xs = []
    plot_ys = []

    # 3) base alphas: first time to reach alpha * oracle
    for alpha in alphas_base:
        target = alpha * oracle_best
        found = None
        for t, b in zip(times, best_perfs):
            if b >= target:
                found = t
                break
        if found is not None:
            plot_xs.append(max(found, 1.0))
            plot_ys.append(alpha)

    # 4) tail: add every improvement after reaching tail_start
    tail_target = tail_start * oracle_best

    # find first index where we reach tail_target
    start_idx = None
    for i, b in enumerate(best_perfs):
        if b >= tail_target:
            start_idx = i
            break

    if start_idx is not None:
        # avoid duplicating the last base point (usually at 0.99)
        last_base_t = plot_xs[-1] if plot_xs else None

        for t, b in zip(times[start_idx:], best_perfs[start_idx:]):
            ratio = min(float(b / oracle_best), 1.0)
            safe_t = max(float(t), 1.0)

            # skip exact duplicate time point (common when first tail point == 0.99 hit)
            if last_base_t is not None and abs(safe_t - last_base_t) < 1e-9 and ratio <= tail_start + 1e-12:
                continue

            plot_xs.append(safe_t)
            plot_ys.append(ratio)

    return plot_xs, plot_ys


def plot_attainment_on_ax(
    ax,
    results_dict,
    recall_min=None,
    qps_min=None,
    tuning_budget=TUNING_BUDGET,
    max_perf=None, # Oracle Performance
    show_xlabel=False,
    show_ylabel=False
):
    """
    Draws an Attainment Plot.
    Y: Target Percentage Levels (0.5, 0.7, ..., 0.99)
    X: Time to Hit (Log Scale)
    """
    # Feasibility Check Factory
    is_feasible, objective, _ = _feasible_and_objective_factory(recall_min, qps_min, tuning_budget, max_perf)

    # Oracle Performance Check
    if max_perf is None or max_perf <= 0:
        ax.text(0.5, 0.5, "No Oracle", ha='center', va='center', fontsize=8)
        return

    # Plot Each Solution
    for sol in SOL_ORDER:
        data = results_dict.get(sol, [])
        style = SOL_STYLES[sol]

        # Calculate t_hits for [0.5, ..., 0.99]
        t_hits = calculate_attainment_times(data, is_feasible, objective, max_perf, tuning_budget)
        
        # Plotting Data Points
        plot_xs = []
        plot_ys = []
        unreached_ys = []
        
        for t, alpha in zip(t_hits, ALPHAS):
            if t is not None:
                # 1초 미만은 1초로 간주 (Log scale 시각화 편의상)
                safe_t = max(t, 1.0)
                plot_xs.append(safe_t)
                plot_ys.append(alpha)
            else:
                unreached_ys.append(alpha)

        # 1. 도달한 점들 그리기 (Line + Marker)
        if plot_xs:
            ax.plot(
                plot_xs, 
                plot_ys, 
                marker=style["marker"], 
                color=style["c"], 
                linestyle=style["ls"], 
                linewidth=style["lw"], 
                markersize=5,
                zorder=style["zorder"],
                label=style["label"],
                alpha=0.9
            )
        
        # # 2. 도달 못한 점들 표시 (Budget 위치에 'x' 마커)
        # if unreached_ys:
        #     ax.scatter(
        #         [tuning_budget * 1.2] * len(unreached_ys), # Budget 약간 뒤에 표시
        #         unreached_ys,
        #         marker="x",
        #         color=style["c"],
        #         s=30,
        #         zorder=style["zorder"],
        #         alpha=0.6
        #     )

    # --- Formatting ---
    # X축: Log Scale (시간)
    ax.set_xscale("log")
    # ax.set_xlim(1.0, tuning_budget * 1.8) # Unreached 마커를 위해 여유 공간 확보
    ax.set_xlim(1.0, tuning_budget * 1.05) # Unreached 마커를 위해 여유 공간 확보
    
    # X축 눈금 설정 (10의 거듭제곱)
    major_ticks = [10**i for i in range(0, 6) if 10**i <= tuning_budget]
    ax.set_xticks(major_ticks)
    
    # 숫자 포맷팅 (10^1 -> 10, 10^2 -> 100)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.get_xaxis().set_major_formatter(formatter)

    if show_xlabel:
        ax.tick_params(axis='x', labelsize=10)
    else:
        ax.set_xticklabels([]) # 라벨 숨기기

    # Y축: Alpha Levels
    # --- [수정됨] 0.5가 추가되었으므로 범위를 0.45부터 시작하도록 조정 ---
    ax.set_ylim(0.7, 1.02)
    ax.set_yticks(ALPHAS)
    
    if show_ylabel:
        ax.set_yticklabels([f"{int(a*100)}%" for a in ALPHAS], fontsize=9)
    else:
        ax.set_yticklabels([]) # 라벨 숨기기
    
    # Grid
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='gray', alpha=0.6)
    
    # 테두리 정리
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Budget Line (Vertical)
    ax.axvline(tuning_budget, linestyle=":", color="gray", linewidth=1.0)


def main():
    # Font Setup
    font_path = f"{current_dir}/LinLibertine_R.ttf"
    if font_path not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "optuna", "nsga"]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = ["nytimes-256-angular", "glove-100-angular", "sift-128-euclidean", "youtube-1024-angular", "deep1M-256-angular"]
    SAMPLING_COUNT = 10
    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"

    # --- Data Loading ---
    results_list = []
    # Row order: hnswlib(recall), hnswlib(qps), faiss(recall), faiss(qps)
    for impl in IMPLS:
        # recall row
        for dataset in DATASETS:
            results_list.append(get_results(impl, dataset, SOLUTIONS, RECALL_MIN, None, SAMPLING_COUNT))
        # qps row
        for dataset in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
            results_list.append(get_results(impl, dataset, SOLUTIONS, None, qps_min, SAMPLING_COUNT))

    # ---- Layout Setup ----
    fig = plt.figure(figsize=(20, 13)) 
    
    gs = gridspec.GridSpec(
        nrows=4, ncols=5, figure=fig,
        left=0.06, right=0.98,
        top=0.90, bottom=0.10,
        wspace=0.15, hspace=0.35 # 간격 조금 줄임
    )

    axes = []
    for r in range(4):
        row_axes = []
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            row_axes.append(ax)
        axes.append(row_axes)

    # ---- Plotting ----
    for idx, rdict in enumerate(results_list):
        r = idx // 5
        c = idx % 5
        ax = axes[r][c]

        # [핵심] 맨 아래 행(r==3)만 x축 라벨 표시, 맨 왼쪽 열(c==0)만 y축 라벨 표시
        show_x = (r == 3)
        show_y = (c == 0)

        plot_attainment_on_ax(
            ax,
            results_dict=rdict["results"],
            recall_min=rdict["recall_min"],
            qps_min=rdict["qps_min"],
            tuning_budget=TUNING_BUDGET,
            max_perf=rdict["oracle_best_metric"],
            show_xlabel=show_x,
            show_ylabel=show_y
        )

        if r == 0:
            ax.set_title(short_ds(rdict["dataset"]), fontsize=18, pad=10, weight='bold')

    # ---- Global Legend ----
    from matplotlib.lines import Line2D
    
    # Custom Legend
    legend_elements = []
    for sol in SOL_ORDER:
        s = SOL_STYLES[sol]
        legend_elements.append(
            Line2D([0], [0], color=s["c"], marker=s["marker"], ls=s["ls"], lw=s["lw"], label=s["label"])
        )
    # legend_elements.append(
    #     Line2D([0], [0], color='gray', marker='x', linestyle='None', label='Not Reached')
    # )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=7, 
        fontsize=15,
        frameon=False,
    )

    # ---- Row Labels & Axis Labels ----
    fig.text(0.015, 0.71, "hnswlib", va="center", ha="center", rotation="vertical", fontsize=22, weight='bold')
    fig.text(0.015, 0.29, "faiss", va="center", ha="center", rotation="vertical", fontsize=22, weight='bold')
    
    label_opts = {'va': 'center', 'ha': 'center', 'rotation': 'vertical', 'fontsize': 13, 'weight': 'bold'}
    fig.text(0.035, 0.815, "Target % (QPS)\n(Recall≥0.95)", **label_opts)
    fig.text(0.035, 0.605, "Target % (Recall)\n(QPS≥Q75)", **label_opts)
    fig.text(0.035, 0.395, "Target % (QPS)\n(Recall≥0.95)", **label_opts)
    fig.text(0.035, 0.185, "Target % (Recall)\n(QPS≥Q75)", **label_opts)
    
    # Common X-Axis Label
    fig.text(0.52, 0.03, "Time to Attain Target (seconds, log scale)", ha='center', fontsize=18)

    output_filename = "main_figure_attainment.pdf"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved attainment plot to {output_filename}")

if __name__ == "__main__":
    main()