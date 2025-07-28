import pandas as pd
import ast

IMPLS = [
    "hnswlib",
    "faiss",
]
DATASETS = [
    "nytimes-256-angular",
    "glove-100-angular",
    "sift-128-euclidean",
    "deep1M-256-angular",
    "youtube-1024-angular",
]

def load_results(file_path):
    df = pd.read_csv(file_path)

    recall_list = []
    qps_list = []

    for _, row in df.iterrows():
        recalls = ast.literal_eval(row["Recall"])
        qps = ast.literal_eval(row["QPS"])
        recall_list.extend(recalls)
        qps_list.extend(qps)

    return recall_list, qps_list

def analyze_results(recall_qps_dict):
    import numpy as np

    print("=" * 60)
    print("Analysis of Recall/QPS stability across test iterations")
    print("=" * 60)

    for key, value in recall_qps_dict.items():
        recall = np.array(value["recall"])
        qps = np.array(value["qps"])

        recall_mean = np.mean(recall)
        recall_std = np.std(recall)
        recall_cv = recall_std / recall_mean if recall_mean != 0 else 0

        qps_mean = np.mean(qps)
        qps_std = np.std(qps)
        qps_cv = qps_std / qps_mean if qps_mean != 0 else 0

        print(f"\n[{key}]")
        print(f"  Recall  -> mean: {recall_mean:.6f}, std: {recall_std:.6f}, CV: {recall_cv:.4%}")
        print(f"  QPS     -> mean: {qps_mean:.2f}, std: {qps_std:.2f}, CV: {qps_cv:.4%}")

        if recall_cv < 0.05:
            print("  ✅ Recall is stable across test iterations.")
        else:
            print("  ⚠️ Recall shows noticeable variation.")

        if qps_cv < 0.05:
            print("  ✅ QPS is stable across test iterations.")
        else:
            print("  ⚠️ QPS shows noticeable variation.")

    print("  --- Stability per number of test iterations ---")
    max_tests = 10  # your data has 10 tests

    for metric_name, data in [("Recall", recall), ("QPS", qps)]:
        print(f"    [{metric_name}]")
        for n in range(1, max_tests + 1):
            means = []
            cvs = []
            # 원본 데이터는 모든 구성의 결과가 합쳐져 있으므로
            # 하나의 구성 당 10개씩 슬라이스해서 봐야 함
            num_configs = len(data) // max_tests

            partial_values = []
            for i in range(num_configs):
                start = i * max_tests
                end = start + n  # 현재 횟수만큼만 사용
                partial = data[start:end]
                partial_values.extend(partial)

            mean = np.mean(partial_values)
            std = np.std(partial_values)
            cv = std / mean if mean != 0 else 0
            print(f"      {n} tests -> mean: {mean:.6f}, CV: {cv:.4%}")

def main():
    recall_qps_dict = {}
    for impl in IMPLS:
        for dataset in DATASETS:
            file_path = f"./data/ground_truths/{impl}/{dataset}.csv"
            try:
                recall_list, qps_list = load_results(file_path)
                recall_qps_dict[f"{impl}_{dataset}"] = {
                    "recall": recall_list,
                    "qps": qps_list
                }
            except FileNotFoundError:
                print(f"File not found: {file_path}")

    analyze_results(recall_qps_dict)

if __name__ == "__main__":
    main()
