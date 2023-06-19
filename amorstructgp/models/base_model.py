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
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def reset_model(self):
        """
        resets a model to internal states that were present at initialization (for example trained parameters)
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, x_data: np.array, y_data: np.array):
        """
        Performs inference of the model - trains all parameters and latent variables needed for prediction

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data: Label array with shape (n,m) where n is the number of training points and m the number of outputs
        """
        raise NotImplementedError

    @abstractmethod
    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape either (n,) if m==1 or (n,m) otherwise where m is the number of output dimensions
        sigma array with shape either (n,) if m==1 or (n,m,m) otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        """
        Method for estimating the model evidence (as we only use bayesian models this should in principle be possible)

        Arguments:
        x_data: Optional - Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data: Optional - Label array with shape (n,1) where n is the number of training points

        Returns:
        evidence value - single value
        """
        raise NotImplementedError

    @abstractmethod
    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
        raise NotImplementedError

    @abstractmethod
    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        """
        Method for calculating the log likelihood value of the the predictive distribution at the test input points (evaluated at the output values)
        - method is therefore for validation purposes only

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points
        y_test: Array of test output points with shape (n,m) where m is the number of output dimensions

        Returns:
        array of shape (n,) with log liklihood values
        """
        raise NotImplementedError
