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
from amorstructgp.utils.plotter2D import Plotter2D


class Warper:
    def __init__(self):
        self.warp_probability = 0.5
        self.rotation_probability = 0.5

    def random_rotation(self, X):
        rotation_matrix = np.random.randn(X.shape[1], X.shape[1])
        rotation_matrix, _ = np.linalg.qr(rotation_matrix)
        rotated_X = np.dot(X, rotation_matrix)
        return rotated_X

    def map_to_unit_interval(self, X):
        min_values = np.min(X, axis=0)
        max_values = np.max(X, axis=0)
        transformed_X = (X - min_values) / (max_values - min_values)
        return transformed_X

    def warp(self, X):
        warp_alphas = np.random.uniform(0.3, 1.0, size=X.shape[1])
        warped_X = np.power(X, warp_alphas)
        return warped_X

    def apply(self, X):
        X = self.map_to_unit_interval(X)
        if np.random.uniform() < self.warp_probability:
            X = self.warp(X)
            X = self.map_to_unit_interval(X)
        if np.random.uniform() < self.rotation_probability:
            X = self.random_rotation(X)
            X = self.map_to_unit_interval(X)
        return X

    def apply_on_batch(self, X_batch):
        transformed_batch = np.array([self.apply(X) for X in X_batch])
        return transformed_batch


if __name__ == "__main__":
    pass
