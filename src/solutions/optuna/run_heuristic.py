import random
from typing import List, Tuple, Optional

import optuna

from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from src.constants import (
    EFC_MAX,
    EFC_MIN,
    EFS_MAX,
    EFS_MIN,
    M_MAX,
    M_MIN,
    IMPL,
    DATASET,
    SEED,
    TUNING_BUDGET,
    RECALL_MIN,
    MAX_SAMPLING_COUNT,
)

# Large penalty used for infeasible configurations
INFEASIBLE_PENALTY = 1e9

STEP_M = 1
STEP_EFC = 8
STEP_EFS = 16


def run(
    impl: str = IMPL,
    dataset: str = DATASET,
    recall_min: Optional[float] = None,
    qps_min: Optional[float] = None,
    tuning_budget: float = TUNING_BUDGET,
    sampling_count: Optional[int] = MAX_SAMPLING_COUNT,
    env=(TUNING_BUDGET, SEED),
):
    assert (recall_min is None) != (qps_min is None), \
        "Only one of recall_min or qps_min should be set."

    random.seed(SEED)

    # GroundTruth instance accumulates tuning_time internally
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)

    # Results buffer compatible with print_optimal_hyperparameters / postprocess_results
    results: List[
        Tuple[
            Tuple[int, int, int],
            Tuple[float, float, float, float, float, float],
        ]
    ] = []

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def snapped_efc_min(M: int) -> int:
        """Return the smallest efC on the grid {EFC_MIN + k*STEP_EFC}
        such that efC >= max(M, EFC_MIN)."""
        raw_min = max(M, EFC_MIN)
        # how many steps of STEP_EFC above EFC_MIN we need
        offset = (raw_min - EFC_MIN + STEP_EFC - 1) // STEP_EFC
        efc_min_bound = EFC_MIN + offset * STEP_EFC
        return efc_min_bound

    def objective(trial: optuna.Trial) -> float:
        # 1) Sample only M, efC on the same grid as grid_search(_heuristic)
        M = trial.suggest_int("M", M_MIN, M_MAX, step=STEP_M)

        efc_min_bound = snapped_efc_min(M)
        if efc_min_bound > EFC_MAX:
            # No valid efC on the grid for this M
            return INFEASIBLE_PENALTY

        efC = trial.suggest_int(
            "efConstruction",
            efc_min_bound,
            EFC_MAX,
            step=STEP_EFC,
        )

        # 2) For this (M, efC), find efS via GroundTruth.get_efS
        #    This mirrors grid_search_heuristic behavior.
        efS = gd.get_efS(
            M,
            efC,
            target_recall=recall_min,
            target_qps=qps_min,
        )

        # get_efS returns 0 if it cannot satisfy the constraint
        # In that case, treat as infeasible.
        if efS == 0:
            # Budget check after get_efS (its internal gd.get calls already consumed time)
            if gd.tuning_time > tuning_budget:
                trial.study.stop()
            return INFEASIBLE_PENALTY

        # 3) Check global tuning budget *after* the efS search
        if gd.tuning_time > tuning_budget:
            trial.study.stop()
            return INFEASIBLE_PENALTY

        # 4) Evaluate performance at the chosen (M, efC, efS)
        #    This should hit the cache and not add extra tuning_time.
        recall, qps, total_time, build_time, index_size = gd.get(
            M=M,
            efC=efC,
            efS=efS,
        )

        # 5) Record the result for logging / postprocessing
        results.append(
            (
                (M, efC, efS),
                (gd.tuning_time, recall, qps, total_time, build_time, index_size),
            )
        )

        ####################################
        # Case 1: Recall constraint is set #
        ####################################
        if recall_min is not None:
            if recall < recall_min:
                # Penalty with distance from feasibility
                return INFEASIBLE_PENALTY + (recall_min - recall)
            # Objective: maximize QPS -> minimize negative QPS
            return -qps

        #################################
        # Case 2: QPS constraint is set #
        #################################
        assert qps_min is not None
        if qps < qps_min:
            return INFEASIBLE_PENALTY + (qps_min - qps)
        # Objective: maximize Recall -> minimize negative Recall
        return -recall

    study.optimize(objective)

    return results


def recall_min():
    """
    Baseline: maximize QPS under Recall >= threshold.
    Mirrors gridsearch_heuristic.recall_min but uses Optuna instead of exhaustive grid.
    """
    for RECALL_MIN in [0.90, 0.95, 0.975]:
        for IMPL in ["hnswlib", "faiss"]:
            for DATASET in [
                "nytimes-256-angular",
                "sift-128-euclidean",
                "glove-100-angular",
                "dbpediaentity-768-angular",
                "msmarco-384-angular",
                "youtube-1024-angular",
            ]:
                results = run(
                    impl=IMPL,
                    dataset=DATASET,
                    recall_min=RECALL_MIN,
                    tuning_budget=TUNING_BUDGET,
                    sampling_count=MAX_SAMPLING_COUNT,
                )
                print_optimal_hyperparameters(
                    results,
                    recall_min=RECALL_MIN,
                )
                postprocess_results(
                    results,
                    solution="optuna",
                    impl=IMPL,
                    dataset=DATASET,
                    recall_min=RECALL_MIN,
                    tuning_budget=TUNING_BUDGET,
                )


def qps_min():
    """
    Baseline: maximize Recall under QPS >= threshold.
    Mirrors gridsearch_heuristic.qps_min but uses Optuna instead of exhaustive grid.
    """
    for QPS_MIN in [2500, 5000, 10000, 25000]:
        for IMPL in ["hnswlib", "faiss"]:
            for DATASET in [
                "nytimes-256-angular",
                "sift-128-euclidean",
                "glove-100-angular",
                "dbpediaentity-768-angular",
                "msmarco-384-angular",
                "youtube-1024-angular",
            ]:
                results = run(
                    impl=IMPL,
                    dataset=DATASET,
                    qps_min=QPS_MIN,
                    tuning_budget=TUNING_BUDGET,
                    sampling_count=MAX_SAMPLING_COUNT,
                )
                print_optimal_hyperparameters(
                    results,
                    qps_min=QPS_MIN,
                )
                postprocess_results(
                    results,
                    solution="optuna",
                    impl=IMPL,
                    dataset=DATASET,
                    qps_min=QPS_MIN,
                    tuning_budget=TUNING_BUDGET,
                )


if __name__ == "__main__":
    recall_min()
    qps_min()
