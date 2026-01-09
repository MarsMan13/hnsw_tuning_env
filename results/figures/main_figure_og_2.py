import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm # cm 모듈 추가

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    # plot_accumulated_timestamp_on_ax, # 여기서 임포트하지 않고 아래에서 재정의합니다.
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
MOCK_SEED = "0_cherry"

# ==========================================
# 1. 스타일 정의 (선 모양, 두께 조정)
# ==========================================
SOL_STYLES = {
    # CHAT (Our Solution): 실선(-), 두께 3.0, 빨간색
    "our_solution":  {"c": "#d62728", "marker": "s", "ls": "-",  "lw": 2.0, "zorder": 10},
    
    # Oracle: 점선(--), 두께 1.5, 파란색
    "brute_force":   {"c": "#1f77b4", "marker": "o", "ls": "--", "lw": 1.5, "zorder": 9},
    
    # Others: 점선(--), 두께 1.8
    "random_search": {"c": "#ff7f0e", "marker": "^", "ls": "--", "lw": 1.2, "zorder": 2},
    "grid_search":   {"c": "#2ca02c", "marker": "D", "ls": "--", "lw": 1.2, "zorder": 1},
    "vd_tuner":      {"c": "#9467bd", "marker": "p", "ls": "--", "lw": 1.2, "zorder": 5},
    "optuna":        {"c": "#8c564b", "marker": "*", "ls": "--", "lw": 1.2, "zorder": 4},
    "nsga":          {"c": "#e377c2", "marker": "x", "ls": "--", "lw": 1.2, "zorder": 3},
}

# ==========================================
# 2. Plot 함수 재정의 (스타일 적용)
# ==========================================
def plot_accumulated_timestamp_on_ax(
    ax,
    results,
    recall_min=None,
    qps_min=None,
    max_perf=None,
    tuning_budget=TUNING_BUDGET,
):
    """
    Plot accumulated timestamp data onto a given Axes using SOL_STYLES.
    """
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."

    for solution, result in results.items():
        if recall_min:
            filtered = [
                (value[0], value[2] if max_perf >= value[2] else max_perf)  # (T_record, qps)
                for _, value in result
                if value[1] >= recall_min
            ]
        elif qps_min:
            filtered = [
                (value[0], value[1] if max_perf >= value[1] else max_perf)  # (T_record, recall)
                for _, value in result
                if value[2] >= qps_min
            ]
        
        if not filtered:
            continue

        filtered.insert(0, (0.0, 0.0))
        filtered.sort(key=lambda x: x[0])

        accumulated = [(0.0, 0.0)]
        current_max = 0.0
        for t, perf in filtered:
            if perf > current_max:
                accumulated.append((t, current_max))
                current_max = perf
                accumulated.append((t, perf))
        
        # 마지막 예산 시점까지 라인 연장
        accumulated.append((tuning_budget, current_max))
        timestamps, perf_values = zip(*accumulated)

        # 라벨 설정
        if solution == "brute_force":
            _label = "Oracle"
        elif solution == "our_solution":
            _label = "CHAT"
        else:
            _label = solution

        # 스타일 가져오기 (없으면 기본값)
        style = SOL_STYLES.get(solution, {"c": "gray", "marker": ".", "ls": "--", "lw": 1.0, "zorder": 1})

        ax.plot(
            timestamps,
            perf_values,
            marker=style["marker"],
            color=style["c"],
            linestyle=style["ls"], # 점선/실선 적용
            linewidth=style["lw"], # 두께 적용
            label=_label,
            markersize=4 if solution != "grid_search" else 3,
            zorder=style["zorder"]
        )

    ax.set_xlim(0, tuning_budget)
    ax.grid(True, linestyle=':', alpha=0.6)


def get_results(
    impl: str,
    dataset: str,
    solutions: list,
    recall_min: float = None,
    qps_min: int = None,
    sampling_count: int = None,
    tuning_time: int = TUNING_BUDGET,
):
    assert (recall_min is not None) != (qps_min is not None), "Either recall_min or qps_min must be specified, but not both."
    results_combi = {}
    max_perf_val = 0.0 # 초기화 수정

    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)
        
        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            if optimal_hp:
                hp = optimal_hp[0]
                _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
                perf = (0.0, recall, qps, total_time, build_time, index_size)
                results = [(hp, perf)]
                # Max perf 업데이트
                current_val = qps if recall_min else recall
                if current_val > max_perf_val:
                    max_perf_val = current_val
        else:
            results = [r for r in results if r[1][0] <= tuning_time]
        
        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "max_perf": max_perf_val
    }

# ... (Analyze Helper 함수들은 그대로 둠) ...
def analyze_all_results(results_list):
    pass # Analysis 생략 (필요하면 기존 코드 사용)

# =========================
# Main
# =========================
def main():
    # 폰트 설정
    font_path_r = f"{current_dir}/LinLibertine_R.ttf"
    if font_path_r not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path_r)

    font_path_b = f"{current_dir}/LinLibertine_B.ttf" 
    if os.path.exists(font_path_b):
        fm.fontManager.addfont(font_path_b)
        print(f"Bold font loaded: {font_path_b}")
    else:
        print(f"Warning: Bold font not found at {font_path_b}.")

    font_name = fm.FontProperties(fname=font_path_r).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    SOLUTIONS = [
        "brute_force", "our_solution", "grid_search",
        "random_search", "vd_tuner", "optuna", "nsga",
    ]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = [
        "nytimes-256-angular", "glove-100-angular", "sift-128-euclidean",
        "youtube-1024-angular", "deep1M-256-angular",
    ]
    COLUMN_LABELS = ["nytimes", "glove", "sift", "deep1M", "youtube"]

    SAMPLING_COUNT = [10]
    RECALL_MINS = [0.95]
    QPS_MIN_KEY = "q75"

    tasks = []
    for impl in IMPLS:
        for metric in ["recall_min", "qps_min"]:
            for dataset in DATASETS:
                for sampling_count in SAMPLING_COUNT:
                    if metric == "recall_min":
                        for recall_min in RECALL_MINS:
                            tasks.append((impl, dataset, SOLUTIONS, recall_min, None, sampling_count))
                    else:
                        qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
                        tasks.append((impl, dataset, SOLUTIONS, None, qps_min, sampling_count))

    results = []
    for impl, dataset, solutions, recall_min, qps_min, sampling_count in tasks:
        results.append(
            get_results(
                impl=impl,
                dataset=dataset,
                solutions=solutions,
                recall_min=recall_min,
                qps_min=qps_min,
                sampling_count=sampling_count,
            )
        )

    # Plot
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))

    for ax, result in zip(axes.flat, results):
        plot_accumulated_timestamp_on_ax(ax, result["results"], result["recall_min"], result["qps_min"], max_perf=result["max_perf"])

    # Legend 설정
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))

    final_handles = []
    final_labels = []

    # CHAT 맨 앞으로
    chat_key = "CHAT" if "CHAT" in by_label else "our_solution"
    if chat_key in by_label:
        final_handles.append(by_label.pop(chat_key))
        final_labels.append("CHAT (Our solution)")

    # 나머지 추가
    final_handles.extend(by_label.values())
    final_labels.extend(by_label.keys())

    leg = fig.legend(
        final_handles,
        final_labels,
        loc="upper center",
        bbox_to_anchor=(0.47, 1.0),
        ncol=7,
        fontsize=18,
        frameon=False,
    )

    # Bold 처리
    for text in leg.get_texts():
        if text.get_text().startswith("CHAT"):
            text.set_fontweight("bold")

    fig.text(0.02, 0.70, "hnswlib", va="center", ha="center", rotation="vertical", fontsize=24)
    fig.text(0.02, 0.30, "faiss", va="center", ha="center", rotation="vertical", fontsize=24)

    for i, label in enumerate(COLUMN_LABELS):
        axes[3, i].set_xlabel(label, fontsize=20, labelpad=10)

    plt.subplots_adjust(left=0.06, bottom=0.1, top=0.94, wspace=0.3, hspace=0.4)

    fig.savefig("main_figure.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()