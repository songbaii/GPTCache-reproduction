from sklearn.linear_model import LogisticRegression
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import expit

class sigmod_iid:
    def __init__(self, delta):
        self.delta: float = delta
        self.P_c: float = 1.0 - self.delta
        self.epsilon_grid: np.ndarray = np.linspace(1e-6, 1 - 1e-6, 50)
        self.thold_grid: np.ndarray = np.linspace(0, 1, 100)

    def wilson_proportion_ci(self, cdf_estimates, n, confidence):
        """
        Vectorized Wilson score confidence interval for binomial proportions.

        Parameters:
        - k : array_like, number of successes (1,tholds,1)
        - n : array_like, number of trials (1)
        - confidence_level : float, confidence level for the interval (1,1,epsilons)

        Returns:
        - ci_low, ci_upp : np.ndarray, lower and upper bounds of the confidence interval
        """
        k = np.asarray((cdf_estimates * n).astype(int))  # (1, tholds,1)
        n = np.asarray(n)  # 1

        assert np.all((0 <= k) & (k <= n)), "k must be between 0 and n"
        assert np.all(n > 0), "n must be > 0"

        p_hat = k / n  # (1, tholds,1)
        z = norm.ppf(confidence)  # this is single sided # (1,1,epsilons)

        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)) / denom

        ci_low = center - margin
        ci_upp = center + margin

        return ci_low, ci_upp  # (1,tholds,epsilons)

    def decide(self,similarity: float, s_vals: list, c_vals: list) -> str:
        if len(s_vals) < 6:
            return 'explore'
        similarities = np.array(s_vals)
        labels = np.array(c_vals)
        num_positive_samples = np.sum(labels == 1)
        num_negative_samples = np.sum(labels == 0)

        negative_samples = similarities[labels == 0].reshape(-1, 1, 1)
        labels = labels.reshape(-1, 1, 1)
        tholds = self.thold_grid.reshape(1, -1, 1)
        deltap = (
            self.delta * (num_negative_samples + num_positive_samples)
        ) / num_negative_samples

        epsilon = self.epsilon_grid[self.epsilon_grid < deltap].reshape(1, 1, -1)

        cdf_estimate = (
            np.sum(negative_samples < tholds, axis=0, keepdims=True)
            / num_negative_samples
        )  # (1, tholds, 1)
        cdf_ci_lower, cdf_ci_upper = self.wilson_proportion_ci(
            cdf_estimate, num_negative_samples, confidence=1 - epsilon
        )  # (1, tholds, epsilon)

        # adjust for positive samples (1,1,epsilon)
        pc_adjusted = 1 - (deltap - epsilon) / (1 - epsilon)

        t_primes = (
            np.sum(cdf_ci_lower > pc_adjusted, axis=1, keepdims=True) == 0
        ) * 1.0 + (
            1 - (np.sum(cdf_ci_lower > pc_adjusted, axis=1, keepdims=True) == 0)
        ) * self.thold_grid[
            np.argmax(cdf_ci_lower > pc_adjusted, axis=1, keepdims=True)
        ]

        t_prime = np.min(t_primes)
        if similarity <= t_prime:
            return 'explore'
        else:
            return 'exploit'
    