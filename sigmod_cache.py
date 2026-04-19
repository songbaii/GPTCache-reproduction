from sklearn.linear_model import LogisticRegression
import numpy as np
import statsmodels.api as sm

class sigmod_cache:
    def __init__(self):
        self.logistic_regression: LogisticRegression = LogisticRegression(
            penalty=None, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False
        )

    def decide(self,similarity: float, s_vals: list, c_vals: list) -> str:
        if len(s_vals) < 6:
            return 'explore'
        
        t_hat = self._estimate_parameters(s_vals, c_vals)

        if similarity >= t_hat:
            return 'exploit'
        else:
            return 'explore'
    
    def _estimate_parameters(self, s_vals: list, c_vals: list) -> float:
        similarities = np.array(s_vals)
        labels = np.array(c_vals)
        similarities = sm.add_constant(similarities)

        try:
            if len(similarities) != len(labels):
                print(f"len does not match: {len(similarities)} != {len(labels)}")
            self.logistic_regression.fit(similarities, labels)
            intercept, gamma = self.logistic_regression.coef_[0]

            gamma = max(gamma, 1e-6)
            t_hat = -intercept / gamma
            t_hat = float(np.clip(t_hat, 0.0, 1.0))

            return t_hat

        except Exception as e:
            print(f"Logistic regression failed: {e}") # 只有在样本全为正例或全为负例时会失败，此时返回一个默认的阈值
            return 0.86
        