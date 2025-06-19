import random
import sys
import os
import time

from matplotlib.pylab import f
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ground_truths import GroundTruth

    
def test0():
    gt = GroundTruth(impl="hnswlib", dataset="nytimes-256-angular")
    ret = gt.get(M=32, efC=96, efS=1024)    # recall, qps, total_time, build_time, index_size

    for M in [17, 33, 49, 63]:
        ret = gt.get(M=M, efC=100, efS=100)
        print(ret)

    print("=====================")
    for efC in [33, 65, 129, 257, 444]:
        ret = gt.get(M=32, efC=efC, efS=100)
        print(ret)
        
def test1():
    """
    Test efS_G2 <= efS_G1
    """
    random.seed(time.time())
    gd = GroundTruth(impl="hnswlib", dataset="nytimes-256-angular")
    ITER = 300
    fail_count = 0
    for iter in range(ITER):
        M_1 = random.randint(4, 64)
        efC_1 = random.randint(8, 512)
        M_2 = M_1 + 2 
        if M_2 > 64:
            continue
        
        efS_1 = gd.get_efS(M=M_1, efC=efC_1, target_recall=0.9, tolerance=1e-6, method="binary")
        efS_2 = gd.get_efS(M=M_2, efC=efC_1, target_recall=0.9, tolerance=1e-6, method="binary")
        
        if efS_2 > efS_1:
            fail_count += 1
            print(f"[{iter}] Failed")
            print(f"efS_1: {efS_1}, efS_2: {efS_2}, M_1: {M_1}, efC_1: {efC_1}, M_2: {M_2}, efC_2: {efC_1}")
    print(f"Pass count: {ITER - fail_count}/{ITER}")

def test2():
    """
    Test efS_G2 <= efS_G1
    """
    random.seed(time.time())
    gd = GroundTruth(impl="hnswlib", dataset="nytimes-256-angular")
    iter = 0
    fail_count = 0
    while iter <= 100:
        M_1 = random.randint(4, 64)
        efC_1 = random.randint(8, 512)
        efC_2 = efC_1 + 4 
        if efC_2 > 512:
            continue
        iter += 1
        efS_1 = gd.get_efS(M=M_1, efC=efC_1, target_recall=0.9, tolerance=1e-6, method="binary")
        efS_2 = gd.get_efS(M=M_1, efC=efC_2, target_recall=0.9, tolerance=1e-6, method="binary")
        
        if efS_2 > efS_1:
            fail_count += 1
            print(f"[{iter}] Failed")
            print(f"efS_1: {efS_1}, efS_2: {efS_2}, M_1: {M_1}, efC_1: {efC_1}, M_2: {M_1}, efC_2: {efC_2}")
    print(f"Pass count: {iter - fail_count}/{iter}")
    
def test3():
    """
    Test 2efS_G1 - efS_G0 <= efS_G2 
    """
    random.seed(time.time())
    gd = GroundTruth(impl="hnswlib", dataset="nytimes-256-angular")
    iter = 0
    fail_count = 0
    while iter <= 100:
        M = random.randint(4, 64)
        efC_1 = random.randint(8, 512)
        efC_0 = efC_1 - 2
        efC_2 = efC_1 + 2 
        if efC_2 > 512 or efC_0 < 8:
            continue
        iter += 1
        efS_0 = gd.get_efS(M=M, efC=efC_0, target_recall=0.9, tolerance=1e-6, method="binary")
        efS_1 = gd.get_efS(M=M, efC=efC_1, target_recall=0.9, tolerance=1e-6, method="binary")
        efS_2 = gd.get_efS(M=M, efC=efC_2, target_recall=0.9, tolerance=1e-6, method="binary")
        
        if 2*efS_1 - efS_0 > efS_2:
            fail_count += 1
            print(f"[{iter}] Failed")
            print(f"{abs(2 * efS_1 - efS_0 - efS_2)}")
    print(f"Pass count: {iter - fail_count}/{iter}")

def test4():
    gd = GroundTruth(impl="hnswlib", dataset="nytimes-256-angular")
    M = 32
    for efC in range(8, 512+1, 8):
        efS = gd.get_efS(M=M, efC=efC, target_recall=0.95, tolerance=1e-4, method="binary")
        recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS)
        print(f"efC: {efC}, efS: {efS}, recall: {recall:.4f}, qps: {qps:.4f}")
  
if __name__ == "__main__":
    test4()