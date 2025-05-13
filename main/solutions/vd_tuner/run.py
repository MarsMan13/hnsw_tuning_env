import sys
import os
sys.path.append("..")

from optimizer_pobo_sa import PollingBayesianOptimization
from utils import MockEnv
from main.constants import IMPL, DATASET, SEED, TUNING_BUDGET
from main.utils import save_search_results, plot_timestamp
from static.ground_truths import  GroundTruth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_EFS = False

def run(use_efS=False):
    gd = GroundTruth(impl=IMPL, dataset=DATASET)
    
    knob_path = os.path.join(BASE_DIR, "params/whole_param.json")
    env = MockEnv(model=gd, knob_path=knob_path, tuning_budget=TUNING_BUDGET, use_efS=use_efS)
    model = PollingBayesianOptimization(env, seed=SEED)
    
    model.init_sample()
    try:
        while True:
            model.step()
    except TimeoutError as e:
        print(f"Tuning time out")
    except KeyboardInterrupt as e:
        print(f"Tuning interrupted by Ctrl+C")
    ## END OF TUNING ##
    results = [] 
    for i in range(len(env.X_record)):
        M, efC, efS = env.X_record[i]
        recall, qps = env.Y_record[i]
        _recall, _qps, total_time, build_time, index_size = gd.get(M, efC, efS, tracking_time=False)
        if abs(recall - _recall) > 1e-5 or abs(qps - _qps) > 1e-5:
            raise ValueError(f"Recall or QPS mismatch: {recall}/{_recall}, {qps}/{_qps}")
        results.append(((M, efC, efS), (env.T_record[i], recall, qps, total_time, build_time, index_size)))
    return results

if __name__ == '__main__':
    results = run(use_efS=USE_EFS)
    save_search_results(results, solution="vd_tuner", filename=f"vd_tuner_{IMPL}_{DATASET}_{TUNING_BUDGET//3600}h_{USE_EFS}_{SEED}.csv")     
    plot_timestamp(results, solution="vd_tuner", filename=f"vd_tuner_{IMPL}_{DATASET}_{TUNING_BUDGET//3600}h_{USE_EFS}_{SEED}.png", min_recall=0.95)