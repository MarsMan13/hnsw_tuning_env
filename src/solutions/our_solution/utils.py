from src.constants import EFC_MAX, EFC_MIN, EFS_MAX, EFS_MIN

class EfCGetter:
    def __init__(self):
        self.__data = {}
        
    def put(self, M, efC_left, efC_right):
        self.__data[M] = (efC_left, efC_right)
    
    def get(self, M): # -> (ef_left, ef_right) of M
        efC_min, efC_max = EFC_MIN, EFC_MAX
        for m in self.__data.keys():
            if m <= M:
                efC_max = min(efC_max, self.__data[m][1])
            if m >= M:
                efC_min = max(efC_min, self.__data[m][0])
        return efC_min, efC_max

class EfSGetter:
    def __init__(self):
        self.__data = dict()
    
    def put(self, efC, efS):
        self.__data[efC] = efS
    
    def get(self, efC):    # -> efS of (M, efC)
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
                _, efs1 = filtered_sorted_items[i]
                _, efs2 = filtered_sorted_items[j]
                efS_max = min(efs2, efS_max)
                efS_min = max(2*efs2 - efs1, efS_min)
        if efS_min != _efS_min or efS_max != _efS_max:
            print(f"({_efS_min}, {_efS_max}) -> ({efS_min}, {efS_max})")
        return efS_min, efS_max