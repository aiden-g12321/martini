"""
Probability distributions.
"""


from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    """
    Abstract probability distribution base class.
    """

    @property
    @abstractmethod
    def ndim(self):
        pass

    @abstractmethod
    def sample(self, size=1):
        """
        Draw samples from the distribution.
        """
        pass

    @abstractmethod
    def logpdf(self, x):
        """
        Evaluate log p(x).
        """
        pass

    def pdf(self, x):
        """
        Evaluate p(x).
        """
        return np.exp(self.logpdf(x))


class Gaussian(Distribution):
    """
    Multivariate Gaussian distribution.
    """

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)

        # Precompute useful quantities
        self.inv_cov = np.linalg.inv(self.cov)
        sign, logdet = np.linalg.slogdet(self.cov)

        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite.")

        self.log_norm = -0.5 * (self.ndim * np.log(2 * np.pi) + logdet)
    
    @property
    def ndim(self):
        return self.mean.shape[0]

    def sample(self, size=1):
        return np.random.multivariate_normal(mean=self.mean,
                                             cov=self.cov,
                                             size=size)

    def logpdf(self, x):
        z = x - self.mean
        mahal = z @ self.inv_cov @ z
        return self.log_norm - 0.5 * mahal


class Cauchy(Distribution):
    """
    1D Cauchy distribution.
    """

    def __init__(self, x0=0.0, gamma=1.0):
        self.x0 = x0
        self.gamma = gamma
    
    @property
    def ndim(self):
        return 1

    def sample(self, size=1):
        return self.x0 + self.gamma * np.random.standard_cauchy(size=size)

    def logpdf(self, x):
        z = (x - self.x0) / self.gamma

        return -np.log(np.pi * self.gamma * (1 + z**2))


class Uniform(Distribution):
    """
    Uniform distribution on [low, high].
    """

    def __init__(self, low=0.0, high=1.0):
        if high <= low:
            raise ValueError("high must be > low")

        self.low = low
        self.high = high
        self.log_density = -np.log(high - low)
    
    @property
    def ndim(self):
        return 1

    def sample(self, size=1):
        return np.random.uniform(
            low=self.low,
            high=self.high,
            size=size,
        )

    def logpdf(self, x):
        x = np.asarray(x)

        inside = (x >= self.low) & (x <= self.high)

        return np.where(inside, self.log_density, -np.inf)

