from src.constants import EFC_MAX, EFC_MIN, EFS_MAX, EFS_MIN

def get_max_perf(results, M, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    if recall_min:
        return max(
            (perf[2] for (m, *_), perf in results if m == M and perf[1] >= recall_min),
            default=0.0
        )
    return max( # if qps_min
        (perf[1] for (m, *_), perf in results if m == M and perf[2] >= qps_min),
        default=0.0
    )

class EfCGetter:
    def __init__(self):
        self.__data = {}
        
    def put(self, M, efC_left, efC_right):
        if M in self.__data:
            existing_left, existing_right = self.__data[M]
            efC_left = max(efC_left, existing_left)
            efC_right = min(efC_right, existing_right)
        else:
            self.__data[M] = (efC_left, efC_right)
    
    def get(self, M): # -> [ef_left, ef_right] of M
        efC_min, efC_max = EFC_MIN, EFC_MAX
        for m in self.__data.keys():
            if m <= M:
                efC_max = min(efC_max, self.__data[m][1])
            if m >= M:
                efC_min = max(efC_min, self.__data[m][0])
        return efC_min, efC_max

    def clear(self):
        self.__data.clear()

    def __contains__(self, M):
        return M in self.__data

class EfSGetter:    # for each M
    def __init__(self):
        self.__data = dict()
    
    def put(self, efC, efS):
        self.__data[efC] = efS
    
    def get(self, efC):    # -> efC's range of efS
        efS_min, efS_max = EFS_MIN, EFS_MAX
        sorted_items = sorted(self.__data.items())
        for efc, efs in sorted_items:
            if efc <= efC:
                efS_max = min(efS_max, efs)
            if efc >= efC:
                efS_min = max(efS_min, efs)
        _efS_min, _efS_max = efS_min, efS_max
        filtered_sorted_items = [(efc, efs) for efc, efs in sorted_items if efc < efC]
        for i in range(len(filtered_sorted_items)):
            for j in range(i+1, len(filtered_sorted_items)):
                efc1, efs1 = filtered_sorted_items[i]
                efc2, efs2 = filtered_sorted_items[j]
                if efC - efc2 <= efc2 - efc1:
                    efS_max = min(efs2, efS_max)
                    efS_min = max(2*efs2 - efs1, efS_min)
        # if efS_min != _efS_min or efS_max != _efS_max:
        #     print(f"({_efS_min}, {_efS_max}) -> ({efS_min}, {efS_max})")
        return efS_min, efS_max
    
    def clear(self):
        self.__data.clear()