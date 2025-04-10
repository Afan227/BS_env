import numpy as np

values = np.arange(0, 220 + 5, 5)
weights = np.where((values <= 60) | (values >= 180), 3, 1)
probabilities = weights / weights.sum()

selected = np.random.choice(values, size=10, p=probabilities)  # 抽 10 个样本
print("Sampled values:", selected)