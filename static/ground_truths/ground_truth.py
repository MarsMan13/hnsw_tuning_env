import ast
import os
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from csv import reader
import numpy as np

from main.constants import EFS_MIN, TOLERANCE
        
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def masked_gaussian_filter(matrix, sigma):
    mask = ~np.isnan(matrix)  # True where value exists
    filled = np.nan_to_num(matrix, nan=0.0)
    
    numerator = gaussian_filter(filled, sigma=sigma)
    denominator = gaussian_filter(mask.astype(float), sigma=sigma)

    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed = np.divide(numerator, denominator)
        smoothed[denominator == 0] = np.nan  # 실제 값 없는 위치는 NaN 유지

    return smoothed


class GroundTruth:
    def __init__(self, impl: str, dataset: str):
        self.impl = impl
        self.dataset = dataset
        self.data = self.load_ground_truths(impl, dataset)
        self.init()
        self.tuning_time = 0.0
        self.current_hp = (0, 0)
        self.searched_cache = dict()    # (M, efC, efS) -> (recall, qps, total_time, build_time, index_size)
        self.searched_timestamp = dict() # (M, efC, efS) -> T_record
        self._get_count = 0
        self.__last_time = 0.0
        
    def init(self, sigma=1.5):
        """
        Initialize interpolators for recall, QPS, total time, and index size,
        using Gaussian smoothing and ground truth data.
        """
        # Step 1: raw data unpacking
        configs = list(self.data.keys())
        points = np.array(configs)  # shape: (N, 3)
        recalls = np.array([v[0] for v in self.data.values()])
        qpss = np.array([v[1] for v in self.data.values()])
        total_times = np.array([v[2] for v in self.data.values()])
        build_times = np.array([v[3] for v in self.data.values()])
        index_sizes = np.array([v[4] for v in self.data.values()])

        # Step 2: smoothing by efS slices
        unique_M = sorted(set(p[0] for p in configs))
        unique_efC = sorted(set(p[1] for p in configs))
        unique_efS = sorted(set(p[2] for p in configs))
        M_index = {m: i for i, m in enumerate(unique_M)}
        efC_index = {e: j for j, e in enumerate(unique_efC)}
        def smooth_metric(metric_values):
            smoothed_values = np.zeros_like(metric_values)
            for efS in unique_efS:
                matrix = np.full((len(unique_M), len(unique_efC)), np.nan)
                mask = []

                for idx, (M, efC, this_efS) in enumerate(configs):
                    if this_efS == efS:
                        i, j = M_index[M], efC_index[efC]
                        matrix[i, j] = metric_values[idx]
                        mask.append((idx, i, j))
                smoothed_matrix = masked_gaussian_filter(matrix, sigma=sigma)
                for idx, i, j in mask:
                    smoothed_values[idx] = smoothed_matrix[i, j]
            return smoothed_values
        
        # Step 3: smooth each metric
        recalls = smooth_metric(recalls)
        qpss = smooth_metric(qpss)
        total_times = smooth_metric(total_times)
        build_times = smooth_metric(build_times)
        index_sizes = smooth_metric(index_sizes)

        # Step 4: interpolators
        self.interp_recall = LinearNDInterpolator(points, recalls)
        self.interp_qps = LinearNDInterpolator(points, qpss)
        self.interp_total_time = LinearNDInterpolator(points, total_times)
        self.interp_build_time = LinearNDInterpolator(points, build_times)
        self.interp_index_size = LinearNDInterpolator(points, index_sizes)
    
    def load_ground_truths(self, impl: str="hnswlib", dataset: str="nytimes"):
        """
        Load the ground truths for a given dataset and implementation.
        Args:
            impl (str): The implementation to use. Defaults to "hnswlib".
            dataset (str): The dataset to use. Defaults to "nytimes".
        Returns:
            dict: A dictionary containing the ground truths.
        """
        results = dict()
        filename = f"{impl}/{dataset}.csv"
        filename = os.path.join(BASE_DIR, filename)
        with open(filename, "r") as f:
            csv_reader = reader(f)
            header = next(csv_reader)
            for row in csv_reader:
                _M = int(row[4])
                _efC = int(row[5])
                _efS = int(row[6])
                _index_size = int(float(row[7]))
                _build_time = float(row[8])
                _total_time = float(row[8]) + float(row[9])
                __recall = ast.literal_eval(row[11])
                _recall = np.array(__recall).mean()
                __qps = ast.literal_eval(row[12])
                _qps = np.array(__qps).mean()
                results[(_M, _efC, _efS)] = (_recall, _qps, _total_time, _build_time, _index_size)
        return results

    def __safe_interp(self, value):
        if value is None or np.isnan(value):
            raise ValueError("Interpolation failed due to NaN or None value.")
        return value

    def get(self, M, efC, efS, tracking_time=True):
        if efS < EFS_MIN:
            return 0.0, 0.0, 0.0, 0.0, 0
        if (M, efC, efS) in self.searched_cache:
            return self.searched_cache[(M, efC, efS)]
        self._get_count += 1
        try:
            recall = self.__safe_interp(self.interp_recall(M, efC, efS))
            qps = self.__safe_interp(self.interp_qps(M, efC, efS))
            total_time = self.__safe_interp(self.interp_total_time(M, efC, efS))
            build_time = self.__safe_interp(self.interp_build_time(M, efC, efS))
            index_size = self.__safe_interp(self.interp_index_size(M, efC, efS))
        except ValueError as e:
            return 0.0, 0.0, 0.0, 0.0, 0
        if tracking_time:
            self.tuning_time += total_time
            if (M, efC) == self.current_hp:
                self.tuning_time -= build_time
            self.searched_cache[(M, efC, efS)] = (recall, qps, total_time, build_time, index_size)
            self.searched_timestamp[(M, efC, efS)] = self.tuning_time
        self.current_hp = (M, efC)
        return recall, qps, total_time, build_time, int(index_size)
    
    def rollback(self):
        self.tuning_time -= (self.tuning_time - self.__last_time)
        self.__last_time = self.tuning_time
    
    def _get_efS_for_qps(self, M, efC, target_qps, method, efS_min, efS_max, tolerance):
        qps_min = self.get(M, efC, efS_min)[1]
        if qps_min < target_qps:
            return 0.0
        best_efS = None
        best_qps = 0.0
        best_diff = float("inf")
        if method == "linear":
            for efS in range(efS_min, efS_max+1):
                recall, qps, *_ = self.get(M, efC, efS)
                if qps == 0.0:
                    continue
                diff = abs(qps - target_qps)
                if diff < best_diff:
                    best_diff = diff
                    best_efS = efS
                    if diff < tolerance:
                        break
        elif method == "binary":
            left, right = efS_min, efS_max
            while left <= right:
                mid = (left + right) // 2
                recall, qps, *_ = self.get(M, efC, mid)
                if qps == 0.0:
                    right = mid - 1
                    continue
                diff = abs(qps - target_qps)
                if diff < best_diff and qps >= target_qps:
                    best_diff = diff
                    best_efS = mid
                    best_qps = qps
                    if diff <= tolerance:
                        break
                if qps < target_qps:
                    right = mid - 1
                else:
                    left = mid + 1
        else:
            raise ValueError(f"Unknown method '{method}'")
        if best_qps < target_qps:
            return 0.0
        return best_efS if best_efS else 0.0        
        
    
    def _get_efS_for_recall(self, M, efC, target_recall, method, efS_min, efS_max, tolerance):
        recall_min = self.get(M, efC, efS_min)[0]
        if recall_min >= target_recall:
            return efS_min
        best_efS = None
        best_recall = 0.0
        best_diff = float("inf")
        if method == "linear":
            for efS in range(efS_min, efS_max+1):
                recall, *_ = self.get(M, efC, efS)
                if recall == 0.0:
                    continue
                diff = abs(recall - target_recall)
                if diff < best_diff:
                    best_diff = diff
                    best_efS = efS
                    if diff < tolerance:
                        break
        elif method == "binary":
            left, right = efS_min, efS_max
            while left <= right:
                mid = (left + right) // 2
                recall, *_ = self.get(M, efC, mid)
                if recall == 0.0:
                    left = mid + 1
                    continue
                diff = abs(recall - target_recall)
                if diff < best_diff and recall >= target_recall:
                    best_diff = diff
                    best_efS = mid
                    best_recall = recall
                    if diff <= tolerance:   #! OPTIMIZED LOGIC
                        break
                if recall < target_recall:
                    left = mid + 1
                else:
                    right = mid - 1
        else:
            raise ValueError(f"Unknown method '{method}'")
        if best_recall < target_recall:
            return 0.0
        return best_efS if best_efS else 0.0
    
    def get_efS(self, M, efC, target_recall=None, target_qps=None, method="binary", efS_min=32, efS_max=1024, tolerance=TOLERANCE, skip_time=False):
        if target_recall is None and target_qps is None:
            raise ValueError("Either target_recall or target_qps must be provided.")
        if target_recall and target_qps:
            raise ValueError("Only one of target_recall or target_qps can be provided.")
        # END OF VALIDATION
        if target_recall:
            return self._get_efS_for_recall(M, efC, target_recall, method, efS_min, efS_max, tolerance)
        if target_qps:
            return self._get_efS_for_qps(M, efC, target_qps, method, efS_min, efS_max, tolerance)
        raise ValueError("Critical error: this should never happen.") 
        
        
if __name__ == "__main__":
    gd = GroundTruth("faiss", "nytimes-256-angular")
    print(gd.get(48, 202, 32))