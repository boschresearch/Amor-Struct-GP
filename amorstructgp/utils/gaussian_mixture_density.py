# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Matthias Bitzer, matthias.bitzer3@de.bosch.com
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from scipy import integrate
from enum import Enum


class EntropyApproximation(Enum):
    QUADRATURE = 1
    FIRST_ORDER_TAYLOR = 2
    SECOND_ORDER_TAYLOR = 3
    SAMPLE = 4
    MOMENT_MATCHED_GAUSSIAN = 5


class GaussianMixtureDensity:
    def __init__(self, weights, mus, sigmas):
        self.weights = weights
        self.mus = mus
        self.sigmas = sigmas
        self.variances = np.power(self.sigmas, 2.0)
        assert self.weights.shape[0] == self.mus.shape[0]
        assert self.weights.shape[0] == self.sigmas.shape[0]
        self.entropy_approx = EntropyApproximation.MOMENT_MATCHED_GAUSSIAN
        self.entropy_sample_size = 1000

    def set_entropy_approx_type(self, entropy_approx, entropy_sample_size=None):
        self.entropy_approx = entropy_approx
        if entropy_sample_size is not None:
            self.entropy_sample_size = entropy_sample_size

    def p(self, x):
        p = 0.0
        for i, weight in enumerate(self.weights):
            p += weight * norm.pdf(x, self.mus[i], self.sigmas[i])
        return p

    def p_diff(self, x):
        p_diff = 0
        for i, weight in enumerate(self.weights):
            p_diff += weight * ((x - self.mus[i]) / self.variances[i]) * norm.pdf(x, self.mus[i], self.sigmas[i])
        return p_diff

    def log_likelihood(self, x: float):
        return np.log(self.p(x))

    def neg_log_likeli(self, x):
        return -1 * np.log(self.p(x))

    def mean(self):
        mean = np.dot(np.squeeze(self.weights), np.squeeze(self.mus))
        return mean

    def variance(self):
        mean = self.mean()
        summands = np.power(self.sigmas, 2.0) + np.power(self.mus, 2.0) - np.power(mean, 2.0)
        assert summands.shape[0] == self.weights.shape[0]
        return np.dot(np.squeeze(self.weights), np.squeeze(summands))

    def plot(self, y=None, add_point=False):
        min_x = np.min(self.mus - 2 * self.sigmas)
        max_x = np.max(self.mus + 2 * self.sigmas)
        xs = np.linspace(min_x - 0.5, max_x + 0.5, 500)
        ps = []
        for x in xs:
            p = self.p(x)
            ps.append(p)
        ps = np.array(ps)
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, ps)
        if add_point:
            ax.plot([y], [0], "x", color="red")
        plt.show()

    def entropy_integrand(self, x):
        p = self.p(x)
        if math.isclose(p, 0.0):
            return 0.0
        else:
            return -1 * p * np.log(p)

    def first_order_taylor_entropy(self):
        value = 0
        for i, weight in enumerate(self.weights):
            value += -1 * weight * np.log(self.p(self.mus[i]))
        return value

    def second_order_taylor_entropy(self):
        value = self.first_order_taylor_entropy()
        for i, weight in enumerate(self.weights):
            value += -0.5 * weight * self.F(self.mus[i]) * self.variances[i]
        return value

    def F(self, x):
        value = 0
        for i, weight in enumerate(self.weights):
            value += (
                (weight / self.variances[i])
                * ((x - self.mus[i]) * (self.p_diff(x) / self.p(x)) + (np.power((x - self.mus[i]), 2.0) / self.variances[i]) - 1)
                * norm.pdf(x, self.mus[i], self.sigmas[i])
            )
        value = value / self.p(x)
        return value

    def draw_categorical(self):
        return np.random.choice(a=list(range(0, len(self.weights))), size=1, p=self.weights)[0]

    def sample_data(self, n):
        samples = []
        for i in range(0, n):
            index = self.draw_categorical()
            sample = norm.rvs(loc=self.mus[index], scale=self.sigmas[index])
            samples.append(sample)
        return np.array(samples)

    def entropy_mc_estimate(self, n_samples):
        entropy = 0
        samples = self.sample_data(n_samples)
        for sample in samples:
            entropy += -1 * np.log(self.p(sample))
        entropy = entropy / len(samples)
        return entropy

    def entropy(self):
        if self.entropy_approx == EntropyApproximation.QUADRATURE:
            f = lambda y: self.entropy_integrand(y)
            lower = min([self.mus[i] - self.sigmas[i] * 2 for i in range(0, len(self.mus))])
            upper = max([self.mus[i] + self.sigmas[i] * 2 for i in range(0, len(self.mus))])
            int_f = integrate.quad(f, lower, upper, limit=30)
            return int_f[0]
        elif self.entropy_approx == EntropyApproximation.FIRST_ORDER_TAYLOR:
            return self.first_order_taylor_entropy()
        elif self.entropy_approx == EntropyApproximation.SECOND_ORDER_TAYLOR:
            return self.second_order_taylor_entropy()
        elif self.entropy_approx == EntropyApproximation.SAMPLE:
            return self.entropy_mc_estimate(self.entropy_sample_size)
        elif self.entropy_approx == EntropyApproximation.MOMENT_MATCHED_GAUSSIAN:
            sigma = np.sqrt(self.variance())
            entropy = np.log(sigma * np.sqrt(2 * np.pi * np.exp(1)))
            return entropy


if __name__ == "__main__":
    variances = [0.2, 0.1, 0.3, 0.05, 0.4]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    means = [0.0, 0.2, 0.7, -0.5, -0.7]
    gmm_marg = GaussianMixtureDensity(np.array(weights), np.array(means), np.sqrt(variances))
    gmm_marg.plot()
    print(gmm_marg.entropy())
