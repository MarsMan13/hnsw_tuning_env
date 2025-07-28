from scipy.stats import spearmanr

x1= [0, 3, 5]
y1 = [0, 5, 5]

x2 = [0, 1, 2, 3, 4, 5]
y2 = [0, 2, 1, 3, 3, 5]

print(spearmanr(x1, y1))
print(spearmanr(x2, y2))