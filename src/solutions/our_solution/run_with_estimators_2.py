import math
import random
from typing import List, Tuple, Dict, Optional

from src.constants import (
    DATASET, IMPL, SEED, TUNING_BUDGET,
    EFC_MIN, EFC_MAX, M_MIN, M_MAX,
    BUILD_TIME_BUDGET, INDEX_SIZE_BUDGET
)
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.our_solution.utils import EfCGetter, EfSGetterV2
from src.solutions.our_solution.stats import Stats
from data.ground_truths.ground_truth import GroundTruth
from src.solutions.our_solution.estimators import BuildTimeEstimator, IndexSizeEstimator, get_data_spec

TERNARY_SEARCH_BASE = 2.5
TERNARY_SEARCH_FACTOR = 3
QPS_PERF_PENALTY = 0.95


HP = Tuple[int, int, int]  # (M, efC, efS)
PerfTuple = Tuple[float, float]  # (recall, qps)
ResultRow = Tuple[HP, Tuple[float, float, float]]  # (hp, (tuning_time, recall, qps))


class HyperparameterTuner:
    """
    Encapsulates the entire hyperparameter tuning process to avoid global state
    and improve code structure.
    """
    def __init__(
        self,
        ground_truth: GroundTruth,
        recall_min: Optional[float] = None,
        qps_min: Optional[float] = None,
        build_time_budget: float = float('inf'),
        index_size_budget: float = float('inf'),
        tuning_budget: float = TUNING_BUDGET,
        log_level: int = 1,
    ):
        assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

        self.ground_truth = ground_truth
        self.recall_min = recall_min
        self.qps_min = qps_min
        self.tuning_budget = tuning_budget

        self.stats: Stats = Stats(tuning_budget=tuning_budget, recall_min=recall_min, qps_min=qps_min)

        # Use dict for O(1) lookup: M -> best performance g(M)
        self.m_to_perf: Dict[int, float] = {}

        # Cache results for O(1) lookup: hp -> (t, recall, qps)
        self.hp_to_result: Dict[HP, Tuple[float, float, float]] = {}

        # Keep insertion order list if you still need original output format
        self.results: List[Tuple[HP, Tuple[float, float, float]]] = []

        self.efC_getter = EfCGetter()
        self.efS_getter = EfSGetterV2()
        ####
        N, d = get_data_spec(ground_truth.dataset)
        self.build_time_budget = build_time_budget
        self.index_size_budget = index_size_budget
        self.__build_count = 0
        self.__log_level = log_level
        if not (1 <= self.__log_level <= 3):
            raise ValueError("log_level must be between 1 and 3.")

    def __log(self, message: str, level: int = 1) -> None:
        """Log a message if current log level is sufficient."""
        if self.__log_level >= level:
            print(message)

    def _objective_value(self, perf: PerfTuple) -> float:
        """
        Return the objective value given (recall, qps).
        - If Recall is constrained, maximize QPS.
        - If QPS is constrained, maximize Recall.
        """
        recall, qps = perf
        return recall if self.qps_min is not None else qps

    def _budget_exhausted(self) -> bool:
        return self.ground_truth.tuning_time >= self.tuning_budget

    def run_tuning(self) -> List[Tuple[HP, Tuple[float, float, float]]]:
        self.__log("--- Starting Exploration Phase ---", level=1)
        self._exploration_phase()
        for result in self.results:
            print(result)
        self.stats.exploration_phase(self.results)
        print_optimal_hyperparameters(self.results, recall_min=self.recall_min, qps_min=self.qps_min)
        self.__log("\n--- Tuning is Done! ---", 1)
        return self.results

    def _exploration_phase(self) -> None:
        """
        Broadly search M space using ternary-search style shrinking.
        """
        m_bottom, m_top = M_MIN, M_MAX
        processed_m = set()

        while (m_top - m_bottom) > 3:
            if self._budget_exhausted():
                self.__log("Budget exceeded during M exploration.", 1)
                return

            m_mid1 = m_bottom + (m_top - m_bottom) // 3
            m_mid2 = m_top - (m_top - m_bottom) // 3

            if m_mid1 not in processed_m:
                self.m_to_perf[m_mid1] = self._find_best_efc_for_m(m_mid1)
                processed_m.add(m_mid1)

            if m_mid2 not in processed_m:
                self.m_to_perf[m_mid2] = self._find_best_efc_for_m(m_mid2)
                processed_m.add(m_mid2)

            perf_mid1 = self.m_to_perf.get(m_mid1, 0.0)
            perf_mid2 = self.m_to_perf.get(m_mid2, 0.0)

            self.__log(
                f"\n[M Exploration] Range [{m_bottom}, {m_top}]: "
                f"M1({m_mid1}) -> {perf_mid1:.4f}, M2({m_mid2}) -> {perf_mid2:.4f}",
                2
            )

            # Heuristic range shrinking (kept behavior, but make it explicit)
            if perf_mid1 == -float('inf') and perf_mid2 == -float('inf'):
                m_top = m_mid1
            elif perf_mid1 == perf_mid2:
                if self.recall_min is not None:
                    m_bottom = m_mid1
                else:
                    m_top = m_mid2
            else:
                perf1 = perf_mid1 * QPS_PERF_PENALTY if self.recall_min is not None else perf_mid1
                if perf1 <= perf_mid2:
                    m_bottom = m_mid1
                else:
                    m_top = m_mid2

        for m_val in range(m_bottom, m_top + 1):
            if self._budget_exhausted():
                return
            if m_val not in processed_m:
                self.m_to_perf[m_val] = self._find_best_efc_for_m(m_val)
                processed_m.add(m_val)

    def _find_best_efc_for_m(self, m: int, is_exploitation: bool = False) -> float:
        """
        Find best efC for a given M via ternary-search style probing plus local sweep.
        """
        efc_left, efc_right = self.efC_getter.get(m)

        if is_exploitation:
            efc_iter_limit = EFC_MAX
        else:
            span = max(EFC_MAX - EFC_MIN, 1)
            efc_iter_limit = max(1, math.ceil(math.log(span, TERNARY_SEARCH_BASE)) // TERNARY_SEARCH_FACTOR)

        best_perf = -float('inf')
        efc_count = 0

        while (efc_right - efc_left) > 3 and efc_count < efc_iter_limit:
            if self._budget_exhausted():
                break

            efc_count += 1
            efc_mid1 = efc_left + (efc_right - efc_left) // 3
            efc_mid2 = efc_right - (efc_right - efc_left) // 3

            perf_mid1 = self._evaluate_hp(m, efc_mid1)
            perf_mid2 = self._evaluate_hp(m, efc_mid2)

            if perf_mid1 == -float('inf') and perf_mid2 == -float('inf'):   # <-- Is it corrected here?
                efc_right = efc_mid1
            elif perf_mid1 == perf_mid2 and self.qps_min is not None:
                efc_right = efc_mid2
                efc_count -= 1
            elif perf_mid1 <= perf_mid2:
                efc_left = efc_mid1
            else:
                efc_right = efc_mid2

            if perf_mid1 != perf_mid2:
                self.efC_getter.put(m, efc_left, efc_right)

            best_perf = max(best_perf, perf_mid1, perf_mid2)
            self.__log(f"[efC Search for M][{m}] {efc_mid1} -> {perf_mid1:.4f}, {efc_mid2} -> {perf_mid2:.4f}", 2)

        for efc in range(efc_left, efc_right + 1):
            if self._budget_exhausted() or efc_count >= efc_iter_limit:
                break

            efc_count += 1
            perf = self._evaluate_hp(m, efc)
            best_perf = max(best_perf, perf)

        self.__log(f"=> Max perf of M={m}: {best_perf:.4f}", 2)
        return best_perf

    def _evaluate_hp(self, m: int, efc: int) -> float:
        """
        Evaluate (M, efC) by selecting efS via GroundTruth and measuring perf.
        Use caching to avoid repeated work.
        """
        if self._budget_exhausted():
            return 0.0

        # Determine efS range and choose efS (may be expensive; avoid if possible)
        efs_min, efs_max = self.efS_getter.get(m, efc)
        efs = self.ground_truth.get_efS(m, efc, self.recall_min, self.qps_min, efS_min=efs_min, efS_max=efs_max)
        self.efS_getter.put(m, efc, efs)

        hp: HP = (m, efc, efs)
        cached = self.hp_to_result.get(hp)
        if cached is not None:
            _, recall, qps = cached
            return self._objective_value((recall, qps))
        recall, qps, total_time, build_time, index_size = self.ground_truth.get(m, efc, efs)
        row = (self.ground_truth.tuning_time, recall, qps)
        self.hp_to_result[hp] = row
        self.results.append((hp, (self.ground_truth.tuning_time, recall, qps, total_time, build_time, index_size)))
        if build_time > self.build_time_budget or index_size > self.index_size_budget:
            return -float('inf')

        return self._objective_value((row[1], row[2]))


def run(
    impl=IMPL,
    dataset=DATASET,
    recall_min=None,
    qps_min=None,
    build_time_budget=float('inf'),
    index_size_budget=float('inf'),
    tuning_budget=TUNING_BUDGET,
    sampling_count=None,
    env=(TUNING_BUDGET, SEED),
    stats=False,
):
    random.seed(SEED)
    ground_truth = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    tuner = HyperparameterTuner(ground_truth, recall_min, qps_min, 
                                build_time_budget, index_size_budget, tuning_budget, log_level=2)
    results = tuner.run_tuning()
    if stats:
        return results, tuner.stats, tuner.efC_getter.stats()
    return results

TEST_CASES = {
    # -------------------------
    # faiss
    # -------------------------
    "faiss-nytimes-q1": (19.49297088384628, 289916.0, "nytimes-256-angular"),
    "faiss-nytimes-q3": (162.01354598999023, 381788.0, "nytimes-256-angular"),

    "faiss-glove-q1": (41.96106892824173, 578820.0, "glove-100-angular"),
    "faiss-glove-q3": (286.3342584967613, 838992.0, "glove-100-angular"),

    "faiss-sift-q1": (33.00829893350601, 598092.0, "sift-128-euclidean"),
    "faiss-sift-q3": (150.76723128557205, 819918.0, "sift-128-euclidean"),

    "faiss-youtube-q1": (144.7929595708847, 4057940.0, "youtube-1024-angular"),
    "faiss-youtube-q3": (743.3625051379204, 4276796.0, "youtube-1024-angular"),

    "faiss-deep1M-q1": (77.45466607809067, 1098504.0, "deep1M-256-angular"),
    "faiss-deep1M-q3": (370.06855714321136, 1322310.0, "deep1M-256-angular"),

    # -------------------------
    # hnswlib
    # -------------------------
    "hnswlib-nytimes-q1": (22.95947790145874, 321173.0, "nytimes-256-angular"),
    "hnswlib-nytimes-q3": (146.99570554494858, 384252.0, "nytimes-256-angular"),

    "hnswlib-glove-q1": (74.6927787065506, 590984.0, "glove-100-angular"),
    "hnswlib-glove-q3": (448.0781384706497, 852802.0, "glove-100-angular"),

    "hnswlib-sift-q1": (46.12476706504822, 609128.0, "sift-128-euclidean"),
    "hnswlib-sift-q3": (228.39782667160034, 828998.0, "sift-128-euclidean"),

    "hnswlib-youtube-q1": (222.86241441965103, 4069453.0, "youtube-1024-angular"),
    "hnswlib-youtube-q3": (1123.1803475618362, 4285987.0, "youtube-1024-angular"),

    "hnswlib-deep1M-q1": (79.21995347738266, 1106490.0, "deep1M-256-angular"),
    "hnswlib-deep1M-q3": (417.98080629110336, 1326168.0, "deep1M-256-angular"),
}


import csv
import os

def save_results_to_csv(results: List[Tuple], filename: str):
    """
    results 리스트를 받아 지정된 형식(M, efC, build_time, index_size)으로 CSV 파일에 저장합니다.
    """
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 헤더 작성
            writer.writerow(["M", "efC", "build_time", "index_size"])
            
            for hp, metrics in results:
                # hp = (M, efC, efS)
                # metrics = (tuning_time, recall, qps, total_time, build_time, index_size)
                
                # 데이터 추출
                m_val = hp[0]
                efc_val = hp[1]
                
                # metrics 튜플 구조에 따라 인덱싱 (build_time: 4, index_size: 5)
                # (tuning_time, recall, qps, total_time, build_time, index_size)
                build_time = metrics[4]
                index_size = metrics[5]
                
                writer.writerow([m_val, efc_val, build_time, index_size])
                
        print(f"Successfully saved results to {filename}")
        
    except Exception as e:
        print(f"Failed to save {filename}: {e}")


def run_recall_min_with_constraints():
    RECALL_MIN = 0.95
    # 테스트를 위해 하나만 지정하거나, 실제로는 TEST_CASES에서 가져온 값을 사용해야 합니다.
    # 현재 코드상 TEST_CASES의 value가 비어있으므로((,)), 실제 실행 시에는 값을 채워야 합니다.

    # 출력 디렉토리 생성 (선택사항)
    os.makedirs("results", exist_ok=True)

    for test_case, (build_time_budget, index_size_budget, DATASET) in TEST_CASES.items():
        IMPL = test_case.split('-')[0]
        print(f"Running test case: {test_case}")
        
        # 1. Build Time Constraint Run
        print(f"  - Build Time Budget: {build_time_budget}")
        build_time_results = run(
            impl=IMPL,
            dataset=DATASET,
            recall_min=RECALL_MIN,
            build_time_budget=build_time_budget,
            index_size_budget=float('inf'),
        )
        
        # 2. Index Size Constraint Run
        print(f"  - Index Size Budget: {index_size_budget}")
        index_size_results = run(
            impl=IMPL,
            dataset=DATASET,
            recall_min=RECALL_MIN,
            build_time_budget=float('inf'),
            index_size_budget=index_size_budget,
        )

        # --- Save CSV Results ---
        # 파일명 생성
        bt_filename = f"{test_case}-build_time.csv"
        is_filename = f"{test_case}-index_size.csv"
        
        # (옵션) results 폴더 안에 저장하고 싶다면 아래 주석 해제
        bt_filename = os.path.join("estimator_results", bt_filename)
        is_filename = os.path.join("estimator_results", is_filename)

        # CSV 저장 함수 호출
        save_results_to_csv(build_time_results, bt_filename)
        save_results_to_csv(index_size_results, is_filename)


if __name__ == "__main__":
    run_recall_min_with_constraints()
