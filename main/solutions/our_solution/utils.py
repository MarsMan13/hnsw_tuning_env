# 1) M -> efC
from main.constants import EFC_MAX, EFC_MIN, EFS_MAX, EFS_MIN


class MToEf:
    def __init__(self):
        self.__data = {}
        
    def put(self, M, ef_left, ef_right):
        self.__data[M] = (ef_left, ef_right)
    
    def get_range(self, M):
        keys = sorted(self.__data.keys())
        le_keys = [k for k in keys if k >= M]
        le_key = min(le_keys) if le_keys else None
        ri_keys = [k for k in keys if k <= M]
        ri_key = max(ri_keys) if ri_keys else None
        le_value, ri_value = EFC_MIN, EFC_MAX
        if le_key:
            le_value = self.__data[le_key][0]
        if ri_key:
            ri_value = self.__data[ri_key][1]
        return le_value, ri_value 
