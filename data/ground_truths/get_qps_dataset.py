from data.ground_truths.ground_truth import GroundTruth
import numpy as np

def get_qps_metrics_dataset(dataset, ret_dict=False):
    gd = GroundTruth(impl="hnswlib", dataset=dataset, sampling_count=10)
    results = gd.load_ground_truths(impl="hnswlib", dataset=dataset)
    QPSs = [qps for _, qps, *__ in results.values()]
    
    mean = np.mean(QPSs)
    median = np.median(QPSs)
    q0 = np.quantile(QPSs, 0.0)
    q25 = np.quantile(QPSs, 0.25)
    q75 = np.quantile(QPSs, 0.75)
    q100 = np.quantile(QPSs, 1.0)
    if not ret_dict:
        return mean, median, q0, q25, q75, q100
    return {
        "mean": mean,
        "median": median,
        "q0": q0,
        "q25": q25,
        "q75": q75,
        "q100": q100
    }

if __name__ == "__main__":
    for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
                    "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
        print(f"\n**** {DATASET} ****")
        for METRIC in ["mean", "median", "q0", "q25", "q75", "q100"]:
            value = get_qps_metrics_dataset(DATASET, METRIC)
            print(f"{METRIC:7}: {value}")
