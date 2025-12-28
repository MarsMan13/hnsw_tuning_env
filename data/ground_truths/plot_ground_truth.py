import os
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from data.ground_truths.ground_truth import GroundTruth
from src.constants import SEED, MAX_SAMPLING_COUNT

from src.constants import M_MIN, M_MAX, EFC_MIN, EFC_MAX, EFS_MIN, EFS_MAX, TOLERANCE
        
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(SEED)

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
    
class PlotGroundTruth(GroundTruth):

    def plot_envelope_4subplots(
        self,
        recall_min: float = None,
        qps_min: float = None,
        outpath: str = "envelope_4plots.png",
    ):
        assert (recall_min is not None) or (qps_min is not None)

        # ===== dense grid (핵심 변경) =====
        Ms = list(range(M_MIN, M_MAX + 1, 2))
        efCs = list(range(EFC_MIN, EFC_MAX + 1, 32))
        X, Y = np.meshgrid(efCs, Ms)

        def best_over_efs(M, efC, mode):
            """
            mode:
            - "qps_under_recall"
            - "recall_under_qps"
            """
            best_perf = -float("inf")
            best_efs = np.nan

            for efS in range(EFS_MIN, EFS_MAX + 1):
                recall, qps, *_ = self.get(M, efC, efS, tracking_time=False)

                # interpolation 실패 or out-of-hull
                if recall == 0.0 and qps == 0.0:
                    continue

                if mode == "qps_under_recall":
                    if recall < recall_min:
                        continue
                    if qps > best_perf:
                        best_perf = qps
                        best_efs = efS

                elif mode == "recall_under_qps":
                    if qps < qps_min:
                        continue
                    if recall > best_perf:
                        best_perf = recall
                        best_efs = efS

            if best_perf < 0:
                return np.nan, np.nan
            return best_perf, best_efs

        # ===== Z matrices =====
        Z_qps_r, Z_efs_qps_r = np.full(X.shape, np.nan), np.full(X.shape, np.nan)
        Z_rec_q, Z_efs_rec_q = np.full(X.shape, np.nan), np.full(X.shape, np.nan)

        for i, M in enumerate(Ms):
            for j, efC in enumerate(efCs):
                if recall_min is not None:
                    perf, efs = best_over_efs(M, efC, "qps_under_recall")
                    Z_qps_r[i, j] = perf
                    Z_efs_qps_r[i, j] = efs

                if qps_min is not None:
                    perf, efs = best_over_efs(M, efC, "recall_under_qps")
                    Z_rec_q[i, j] = perf
                    Z_efs_rec_q[i, j] = efs

        # ===== plotting =====
        fig = plt.figure(figsize=(14, 10))
        plots = [
            (Z_qps_r, f"Max QPS (Recall ≥ {recall_min})", "QPS"),
            (Z_efs_qps_r, f"efS @ Max QPS (Recall ≥ {recall_min})", "efS"),
            (Z_rec_q, f"Max Recall (QPS ≥ {qps_min})", "Recall"),
            (Z_efs_rec_q, f"efS @ Max Recall (QPS ≥ {qps_min})", "efS"),
        ]

        for idx, (Z, title, zlabel) in enumerate(plots, start=1):
            ax = fig.add_subplot(2, 2, idx, projection="3d")
            ax.plot_surface(
                X, Y, Z,
                cmap="viridis",
                linewidth=0,
                antialiased=True
            )
            ax.set_title(title)
            ax.set_xlabel("efC")
            ax.set_ylabel("M")
            ax.set_zlabel(zlabel)
            if zlabel == "Recall":
                # M축과 평행한 시선
                ax.view_init(elev=30, azim=-60)
            elif zlabel == "QPS":
                # 180도 회전
                ax.view_init(elev=30, azim=120)
            else:
                ax.view_init(elev=30, azim=0)

        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close(fig)



def _plot_worker(args):
    impl, dataset, recall_min, qps_min = args
    try:
        gd = PlotGroundTruth(impl, dataset)
        outpath = os.path.join(
            BASE_DIR, f"{impl}_{dataset}_envelope_4plots.png"
        )
        gd.plot_envelope_4subplots(
            recall_min=recall_min,
            qps_min=qps_min,
            outpath=outpath,
        )
        print(f"[DONE] {impl} / {dataset}")
    except Exception as e:
        print(f"[ERROR] {impl} / {dataset}: {e}")


def plot_ground_truths_parallel(
    recall_min=0.95,
    qps_min=10000,
    n_workers=None,
):
    # impls = ["hnswlib", "faiss"]
    # datasets = [
    #     "nytimes-256-angular",
    #     "glove-100-angular",
    #     "sift-128-euclidean",
    #     "deep1M-256-angular",
    #     "youtube-1024-angular",
    # ]
    impls = ["milvus"]
    datasets = [
        "glove-100-angular",
    ]

    tasks = [
        (impl, dataset, recall_min, qps_min)
        for impl in impls
        for dataset in datasets
    ]

    if n_workers is None:
        n_workers = min(cpu_count(), len(tasks))

    print(f"Running {len(tasks)} tasks with {n_workers} workers")

    with Pool(processes=n_workers) as pool:
        pool.map(_plot_worker, tasks)

if __name__ == "__main__":
    plot_ground_truths_parallel(
        recall_min=0.95,
        qps_min=10000,
    )
