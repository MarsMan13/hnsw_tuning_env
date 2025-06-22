def get_max_perf(results, M, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    if recall_min:
        return max(
            (perf for (m, *_), perf in results if m == M and perf[1] >= recall_min),
            default=(0.0, 0.0, 0.0, 0.0, 0)
        )
    return max(
        (perf for (m, *_), perf in results if m == M and perf[2] >= qps_min),
        default=(0.0, 0.0, 0.0, 0.0, 0)
    )

# def get_exploitation_targets(results):
#     from operator import itemgetter
#     # Step 1
#     targets_for_M = {}
#     for (M, efC, efS), (_, _, qps, *_) in results:
#         if qps < 1.0:  continue
#         lst = targets_for_M.setdefault(M, [])
#         lst.append((efC, qps.item(), efS))
#         lst.sort(key=itemgetter(1), reverse=True)
#         if len(lst) > 2:
#             lst.pop()
#     # Step 2
#     filtered = {
#         M: (first, second)
#         for M, (first, second) in targets_for_M.items()
#         if first and second
#     }
#     # Step 3
#     ranked = sorted(
#         [(M, first, second) for M, (first, second) in filtered.items()],
#         key=lambda x: x[1][1],  # first의 qps
#         reverse=True
#     )
#     if not ranked:
#         return []
#     # Step 4 : Linear Score Allocation
#     min_qps = ranked[-1][1][1]
#     max_qps = ranked[0][1][1]
#     normalized_scores = [
#         (first[1] - min_qps) / (max_qps - min_qps)
#         for _, first, _ in ranked
#     ]
#     total_norm = sum(normalized_scores)
#     final = [
#         (score / total_norm, M, first, second)
#         for score, (M, first, second) in zip(normalized_scores, ranked)
#     ]
#     return final

# def find_value_of_dict(d, key, mode):
#     # mode = "max": find first key >= key
#     # mode = "min": find last key < key
#     import bisect
#     keys = sorted(d.keys())

#     if mode == "max":
#         index = bisect.bisect_left(keys, key)
#         if index < len(keys):
#             return d[keys[index]]
#     elif mode == "min":
#         index = bisect.bisect_right(keys, key)
#         if index > 0:
#             return d[keys[index - 1]]
#     else:
#         raise ValueError("mode must be 'max' or 'min'")
    
#     return (None, None)
