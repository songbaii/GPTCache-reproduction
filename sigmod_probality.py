from sklearn.linear_model import LogisticRegression
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

class sigmod_probality:
    def __init__(self, delta):
        self.logistic_regression: LogisticRegression = LogisticRegression(
            C=np.inf, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False
        )
        self.delta = delta
        self.P_c = 1 - delta

    def decide(self,similarity: float, s_vals: list, c_vals: list) -> str:
        if len(s_vals) < 6:
            return 'explore'
        if self.estimate(similarity, s_vals, c_vals) >= self.P_c:
            return 'exploit'
        else:
            return 'explore'

    def estimate(self, similarity_score: float, s_vals: list, c_vals: list) -> float:
        similarities = np.array(s_vals)
        labels = np.array(c_vals)
        similarities = sm.add_constant(similarities)

        try:
            self.logistic_regression.fit(similarities, labels)
            intercept, gamma = self.logistic_regression.coef_[0]

            gamma = max(gamma, 1e-6)
            t_hat = -intercept / gamma
            t_hat = float(np.clip(t_hat, 0.0, 1.0))

            linear_term = intercept + gamma * similarity_score
            exploration_probability_at_similarity_score = expit(linear_term)

            return exploration_probability_at_similarity_score
        except Exception as e:
            print(f"Logistic regression failed: {e}")
            return -1