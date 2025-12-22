import os
import csv
import math
from typing import Dict, Tuple, List, Any

from data.ground_truths.ground_truth import GroundTruth

HP = Tuple[int, int, int]  # (M, efC, efS) as in your printout
Perf = Tuple[float, float, float, float, int]  # (Recall, QPS, ..., ..., ...)


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("inf")


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def summarize_errors(errors: List[float]) -> Dict[str, float]:
    """Return common summary stats for a list of errors."""
    if not errors:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    xs = sorted(errors)
    n = len(xs)

    def pct(p: float) -> float:
        # Nearest-rank percentile (simple and stable)
        k = max(1, int(math.ceil(p * n))) - 1
        return xs[k]

    return {
        "n": float(n),
        "mean": mean(xs),
        "std": std(xs),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": xs[-1],
    }


def load_gt(impl: str, dataset: str, count: int) -> Dict[HP, Perf]:
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=count)
    # gd.data: Dict[hp, perf]
    return gd.data


def run_analysis(
    impls: List[str],
    datasets: List[str],
    counts: List[int],
    ref_count: int = 10,
    out_dir: str = "analysis_outputs",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # For CSV: per-(impl, dataset, hp, count) record
    per_hp_csv_path = os.path.join(out_dir, "per_hp_errors_vs_ref.csv")

    # For CSV: aggregated summaries per (impl, dataset, count)
    summary_csv_path = os.path.join(out_dir, "summary_errors_vs_ref.csv")

    per_hp_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for impl in impls:
        for dataset in datasets:
            # Load reference (count=10) first
            if ref_count not in counts:
                raise ValueError(f"ref_count={ref_count} must be included in counts={counts}")

            ref_data = load_gt(impl, dataset, ref_count)

            # Load other counts
            data_by_count: Dict[int, Dict[HP, Perf]] = {ref_count: ref_data}
            for c in counts:
                if c == ref_count:
                    continue
                data_by_count[c] = load_gt(impl, dataset, c)

            # Find common HP keys across all counts to compare apples-to-apples
            common_hps = set(ref_data.keys())
            for c, d in data_by_count.items():
                common_hps &= set(d.keys())

            common_hps = sorted(common_hps)
            if not common_hps:
                print(f"[WARN] No common HPs for impl={impl}, dataset={dataset}. Skipping.")
                continue

            # Prepare aggregated error buckets per count
            for c in counts:
                if c == ref_count:
                    continue

                recall_abs_errs: List[float] = []
                recall_rel_errs: List[float] = []
                qps_abs_errs: List[float] = []
                qps_rel_errs: List[float] = []

                for hp in common_hps:
                    ref_perf = data_by_count[ref_count][hp]
                    cur_perf = data_by_count[c][hp]

                    ref_recall, ref_qps = ref_perf[0], ref_perf[1]
                    cur_recall, cur_qps = cur_perf[0], cur_perf[1]

                    r_abs = abs(cur_recall - ref_recall)
                    r_rel = abs(safe_div(cur_recall - ref_recall, ref_recall))

                    q_abs = abs(cur_qps - ref_qps)
                    q_rel = abs(safe_div(cur_qps - ref_qps, ref_qps))

                    recall_abs_errs.append(r_abs)
                    recall_rel_errs.append(r_rel)
                    qps_abs_errs.append(q_abs)
                    qps_rel_errs.append(q_rel)

                    per_hp_rows.append(
                        {
                            "impl": impl,
                            "dataset": dataset,
                            "hp_M": hp[0],
                            "hp_efC": hp[1],
                            "hp_efS": hp[2],
                            "count": c,
                            "ref_count": ref_count,
                            "ref_recall": ref_recall,
                            "cur_recall": cur_recall,
                            "abs_err_recall": r_abs,
                            "rel_err_recall": r_rel,
                            "ref_qps": ref_qps,
                            "cur_qps": cur_qps,
                            "abs_err_qps": q_abs,
                            "rel_err_qps": q_rel,
                        }
                    )

                # Summaries for this (impl, dataset, count)
                r_abs_s = summarize_errors(recall_abs_errs)
                r_rel_s = summarize_errors(recall_rel_errs)
                q_abs_s = summarize_errors(qps_abs_errs)
                q_rel_s = summarize_errors(qps_rel_errs)

                summary_rows.append(
                    {
                        "impl": impl,
                        "dataset": dataset,
                        "count": c,
                        "ref_count": ref_count,
                        "n_common_hps": int(r_abs_s["n"]),
                        # Recall error summaries
                        "recall_abs_mean": r_abs_s["mean"],
                        "recall_abs_p50": r_abs_s["p50"],
                        "recall_abs_p90": r_abs_s["p90"],
                        "recall_abs_p95": r_abs_s["p95"],
                        "recall_abs_max": r_abs_s["max"],
                        "recall_rel_mean": r_rel_s["mean"],
                        "recall_rel_p95": r_rel_s["p95"],
                        "recall_rel_max": r_rel_s["max"],
                        # QPS error summaries
                        "qps_abs_mean": q_abs_s["mean"],
                        "qps_abs_p50": q_abs_s["p50"],
                        "qps_abs_p90": q_abs_s["p90"],
                        "qps_abs_p95": q_abs_s["p95"],
                        "qps_abs_max": q_abs_s["max"],
                        "qps_rel_mean": q_rel_s["mean"],
                        "qps_rel_p95": q_rel_s["p95"],
                        "qps_rel_max": q_rel_s["max"],
                    }
                )

            # Console preview (compact)
            print(f"\n=== impl={impl}, dataset={dataset} (ref={ref_count}) ===")
            for row in summary_rows:
                if row["impl"] == impl and row["dataset"] == dataset:
                    c = row["count"]
                    print(
                        f"- count={c}: "
                        f"Recall rel p95={row['recall_rel_p95']:.4f}, "
                        f"QPS rel p95={row['qps_rel_p95']:.4f}, "
                        f"(n={row['n_common_hps']})"
                    )

    # Write CSVs
    with open(per_hp_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_hp_rows[0].keys()) if per_hp_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_hp_rows:
            writer.writerow(r)

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f"\n[Done] Wrote:")
    print(f"- {per_hp_csv_path}")
    print(f"- {summary_csv_path}")


if __name__ == "__main__":
    # IMPLS = ["hnswlib", "faiss"]
    # DATASETS = [
    #     "nytimes-256-angular",
    #     "glove-100-angular",
    #     "sift-128-euclidean",
    #     "youtube-1024-angular",
    #     "deep1M-256-angular",
    # ]
    IMPLS = ["milvus"]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
    ]
    COUNTS = [10, 5, 3, 1]  # evaluation counts to compare
    run_analysis(IMPLS, DATASETS, COUNTS, ref_count=10, out_dir="analysis_outputs")
