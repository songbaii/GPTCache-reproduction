import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.special import expit  # sigmoid 函数

def sigmoid(s, t, gamma):
    """sigmoid 函数:P(correct | similarity)"""
    return expit(gamma * (s - t))

def binary_cross_entropy(params, s_vals, c_vals):
    """论文公式 (10) 的损失函数"""
    t, gamma = params
    
    # gamma 必须为正数，如果优化器给负数，罚分
    if gamma <= 0:
        return 1e10
        
    # 预测概率
    p = sigmoid(s_vals, t, gamma)
    
    # 防止 log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    
    # 交叉熵损失
    loss = -np.mean(c_vals * np.log(p) + (1 - c_vals) * np.log(1 - p))
    return loss

def estimate_params_mle(s_vals, c_vals):
    """
    输入：
        s_vals: list/np.array, 历史相似度分数
        c_vals: list/np.array, 是否命中正确 (0 或 1)
    输出：
        t_hat, gamma_hat
    """
    # 初始猜测：t=0.85 (中间值), gamma=10 (适中的陡峭度)
    initial_guess = [0.85, 10.0]
    
    # 边界限制：
    # t 在 [0, 1] 之间 (相似度范围)
    # gamma 在 [0.1, 50] 之间 (防止过于平缓或陡峭导致数值问题)
    bounds = [(0.0, 1.0), (0.1, 50.0)]
    
    result = minimize(
        binary_cross_entropy, 
        initial_guess, 
        args=(s_vals, c_vals),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    if result.success:
        return result.x[0], result.x[1]
    else:
        raise ValueError("Optimization failed:", result.message)
    
if __name__ == "__main__":
    # ==================== 模拟 vCache 的 O_nn(x) 数据 ====================
    # 假设这是挂在某个特定 embedding 下的历史记录
    # 比如：相似度 0.99 -> 正确(1), 相似度 0.95 -> 正确(1), 相似度 0.85 -> 错误(0)
    samples_s = np.array([0.99, 0.97, 0.95, 0.88, 0.83, 0.80])
    samples_c = np.array([1,    1,    1,    0,    0,    0])
    
    t_opt, gamma_opt = estimate_params_mle(samples_s, samples_c)
    print(f"估计出的阈值 t_hat: {t_opt:.4f}")
    print(f"估计出的陡峭度 gamma_hat: {gamma_opt:.4f}")
    
    # 验证：计算在相似度 0.92 时的缓存命中概率
    test_s = 0.92
    prob = sigmoid(test_s, t_opt, gamma_opt)
    print(f"在相似度 {test_s} 时，预测正确概率为: {prob:.2%}")
