import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Dict
import random

# ============================================================
# 核心算法：vCache 决策引擎
# ============================================================

class SimpleVCache:
    """
    简化版 vCache 决策引擎
    
    使用方法:
        cache = SimpleVCache(delta=0.05)
        
        # 每次请求时调用
        action, metadata = cache.decide(similarity=0.87, metadata=metadata)
        
        # 如果 action 是 "explore"，生成新回复后需要反馈结果
        if action == "explore":
            response = call_llm(prompt)
            cache.update(metadata, similarity=0.87, is_correct=check_correctness(...))
    """
    
    def __init__(self, delta: float = 0.05, min_samples: int = 6):
        """
        Args:
            delta: 用户设定的最大错误率（如 0.05 表示允许 5% 错误率）
            min_samples: 最少需要多少观测样本才能开始利用缓存
        """
        self.delta = delta
        self.P_c = 1.0 - delta           # 目标正确率
        self.min_samples = min_samples
        
        # ε 搜索网格（论文中遍历所有 ε 找最小 τ）
        self.epsilon_grid = np.linspace(1e-6, 1 - 1e-6, 50)
        
        # 逻辑回归模型（用于拟合 sigmoid 曲线）
        self.logistic = LogisticRegression(
            C=1e6,  # 接近无正则化
            solver='lbfgs', 
            max_iter=1000,
            fit_intercept=False
        )
        
        # 完美分离时的方差映射表（工程兜底策略）
        self.variance_map: Dict[int, List[float]] = {
            6: 0.035445,
            7: 0.028285,
            8: 0.026436,
            9: 0.021349,
            10: 0.019371,
            11: 0.012615,
            12: 0.011433,
            13: 0.010228,
            14: 0.009963,
            15: 0.009253,
            16: 0.011674,
            17: 0.013015,
            18: 0.010897,
            19: 0.011841,
            20: 0.013081,
            21: 0.010585,
            22: 0.014255,
            23: 0.012058,
            24: 0.013002,
            25: 0.011715,
            26: 0.00839,
            27: 0.008839,
            28: 0.010628,
            29: 0.009899,
            30: 0.008033,
            31: 0.00457,
            32: 0.007335,
            33: 0.008932,
            34: 0.00729,
            35: 0.007445,
            36: 0.00761,
            37: 0.011423,
            38: 0.011233,
            39: 0.006783,
            40: 0.005233,
            41: 0.00872,
            42: 0.010005,
            43: 0.01199,
            44: 0.00977,
            45: 0.01891,
            46: 0.01513,
            47: 0.02109,
            48: 0.01531,
        }

    
    # ============================================================
    # 公开接口
    # ============================================================
    
    def decide(self, similarity: float, s_vals: List[float], c_vals: List[int]) -> str:
        """
        决定是探索还是利用
        
        Args:
            similarity: 当前查询与缓存条目的相似度
            s_vals: 相似度值列表
            c_vals: correctness 值列表

        Returns:
            action: "explore" 或 "exploit"
        """

        # 样本不足时强制探索
        if len(s_vals) < self.min_samples:
            return "explore"
        
        # 提取历史数据
        similarities = np.array(s_vals)
        labels = np.array(c_vals)
        
        # 估计参数
        t_hat, gamma, var_t = self._estimate_parameters(similarities, labels)
        if t_hat == -1:  # 估计失败
            return "explore"
        
        # 计算探索概率 τ
        tau = self._compute_tau(
            similarity=similarity,
            t_hat=t_hat,
            gamma=gamma,
            var_t=var_t
        )
        
        # 随机决策
        u: float = random.uniform(0, 1)
        if u <= tau:
            return "explore"
        else:
            return "exploit"

    # ============================================================
    # 内部算法
    # ============================================================
    
    def _estimate_parameters(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        用逻辑回归估计参数
        
        Returns:
            t_hat: 估计的阈值
            gamma: 估计的陡峭度
            var_t: t_hat 的方差
        """
        # 添加常数项（对应逻辑回归的 intercept）
        X = np.column_stack([np.ones(len(similarities)), similarities])
        
        try:
            self.logistic.fit(X, labels)
            intercept, gamma = self.logistic.coef_[0]
            
            gamma = max(gamma, 1e-6)
            t_hat = -intercept / gamma
            t_hat = float(np.clip(t_hat, 0.0, 1.0))
            
            # 检查是否完美分离
            perfect_separation = (
                np.min(similarities[labels == 1]) > np.max(similarities[labels == 0])
            )
            
            var_t = self._compute_variance(
                perfect_separation=perfect_separation,
                n_obs=len(similarities),
                X=X,
                gamma=gamma,
                intercept=intercept
            )
            
            return round(t_hat, 3), round(gamma, 3), var_t
            
        except Exception:
            return -1.0, -1.0, -1.0
    
    def _compute_variance(
        self, perfect_separation: bool, n_obs: int, 
        X: np.ndarray, gamma: float, intercept: float
    ) -> float:
        """计算 t_hat 的方差"""
        if perfect_separation:
            # 完美分离时查表
            if n_obs in self.variance_map:
                return self.variance_map[n_obs]
            else:
                return self.variance_map[max(self.variance_map.keys())]
        else:
            # Delta 方法
            p = self.logistic.predict_proba(X)[:, 1]
            W = p * (1 - p)
            H = X.T @ (W[:, None] * X)  # Hessian
            
            cov_beta = np.linalg.inv(H)
            grad = np.array([-1.0 / gamma, intercept / (gamma ** 2)])
            var_t = float(grad @ cov_beta @ grad)
            return max(0.0, var_t)
    
    def _confidence_interval(
        self, t_hat: float, var_t: float, quantile: float
    ) -> float:
        """
        Return the (upper) quantile-threshold t' such that
          P_est( t > t' ) <= 1 - quantile
        Args
            t_hat: float - The estimated threshold
            var_t: float - The variance of t
            quantile: float - The quantile
        Returns
            float - The t_prime value
        """
        z = norm.ppf(quantile)
        t_prime = t_hat + z * np.sqrt(var_t)
        return float(np.clip(t_prime, 0.0, 1.0))

    def _get_t_primes(self, t_hat: float, var_t: float) -> List[float]:
        """
        Compute all possible t_prime values.
        Args
            t_hat: float - The estimated threshold
            var_t: float - The variance of t
        Returns
            List[float] - The t_prime values
        """
        t_primes: List[float] = np.array(
            [
                self._confidence_interval(
                    t_hat=t_hat, var_t=var_t, quantile=(1 - self.epsilon_grid[i])
                )
                for i in range(len(self.epsilon_grid))
            ]
        )
        return t_primes

    def _compute_tau(
        self, similarity: float, t_hat: float, gamma: float, var_t: float
    ) -> float:
        """
        计算最小探索概率 τ
        
        Returns:
            tau: 探索概率
        """
        t_primes: List[float] = self._get_t_primes(t_hat=t_hat, var_t=var_t)
        likelihoods = self._sigmoid(s=similarity, t=t_primes, gamma=gamma)
        alpha_lower_bounds = (1 - self.epsilon_grid) * likelihoods

        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)
        self.global_t_prime = t_primes[np.argmin(taus)]
        return round(np.min(taus), 5)
        
        
    @staticmethod
    def _sigmoid(s: float, t: float, gamma: float) -> float:
        """Sigmoid 函数"""
        return expit(gamma * (s - t))

if __name__ == "__main__":
    # 简单测试
    cache = SimpleVCache(delta=0.1)
    s_vals = [0.9, 0.85, 0.8, 0.95, 0.88, 0.92]
    c_vals = [1, 1, 0, 1, 1, 1]
    action = cache.decide(similarity=0.87, s_vals=s_vals, c_vals=c_vals)
    print(f"Action: {action}")