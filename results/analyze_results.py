from matplotlib.pylab import f
from matplotlib.pyplot import step
from main.utils import load_search_results
from main.solutions import postprocess_results, print_optimal_hyperparameters
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze results of hyperparameter tuning.")
    parser.add_argument("-o", "--optima", action="store_true", default=False,)
    parser.add_argument("-m", "--fixed_M", type=int, default=None)
    return parser.parse_args()

def load_results(solution, impl, dataset, recall_min):
    filename = f"{solution}_{impl}_{dataset}_{recall_min}r.csv"
    results = load_search_results(solution, filename)
    return results

def get_M_results(results, M, step_efC=1):
    filtered_results = []
    for hp, perf in results:
        if hp[0] == M and hp[1] % step_efC == 0:
            filtered_results.append((hp, perf))
    return filtered_results

if __name__ == "__main__":
    args = parse_args() 
    for SOLUTION in ["brute_force"]:
        for IMPL in ["faiss"]:
        # for IMPL in ["faiss"]:
            for DATASET in ["nytimes-256-angular"]:
            # for DATASET in ["glove-100-angular"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
                for RECALL_MIN in [0.90]:
                    results = load_results(SOLUTION, IMPL, DATASET, RECALL_MIN)
                    print(f"[{SOLUTION}] {IMPL} {DATASET} {RECALL_MIN}r")
                    if args.optima:
                        print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                    if args.fixed_M:
                        filtered_results = get_M_results(results, M=args.fixed_M, step_efC=24)
                        print_optimal_hyperparameters(filtered_results, recall_min=RECALL_MIN)
                        for hp, perf in filtered_results:
                            print(f"{hp} | {perf}")