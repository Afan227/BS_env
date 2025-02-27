import numpy as np
from scipy.stats import rice

def rician_fading():
    """
    使用Rician衰落模型来模拟信号的衰落。
    :param distance: 信号传播的距离
    :param K_factor: 直射信号的强度与多径信号的强度比率
    :param sigma: 多径信号的标准差
    :return: 衰落后的信号强度（幅度）
    """
    # ------------------- Rician 衰落模型参数 -------------------
    K = 3  # Rician K 因子
    s = np.sqrt(K / (K + 1))  # 直射路径强度
    sigma = np.sqrt(1 / (2 * (K + 1)))  # 散射信号标准差
    # 生成瑞利衰落的随机部分（代表多径部分）
    fading = rice.rvs(s / sigma, scale=sigma)
    return 10 * np.log10(fading ** 2)

for i in range(100):
    a = rician_fading()
    print(a)