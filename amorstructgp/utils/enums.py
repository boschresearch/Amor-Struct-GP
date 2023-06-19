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
from enum import Enum, IntEnum


class OptimizerType(str, Enum):
    ADAM = "Adam"
    RADAM = "RAdam"


class LossFunctionType(str, Enum):
    NMLL = "NMLL"  # negative (mean) log marginal likelihood -1/B\sum_{i=1}^{B} 1/N_i log p(y_i|X_i,\theta_{i}=g(D,k_i))
    NMLL_PARAM_SCALED = "NMLL_PARAM_SCALED"  # negative (mean) log marginal likelihood -1/B\sum_{i=1}^{B} 1/N_i 1/|\theta_{i}| log p(y_i|X_i,\theta_{i}=g(D,k_i))
    NMLL_SQRT_PARAM_SCALED = "NMLL_SQRT_PARAM_SCALED"  # negative (mean) log marginal likelihood -1/B\sum_{i=1}^{B} 1/N_i 1/\sqrt(|\theta_{i}|) log p(y_i|X_i,\theta_{i}=g(D,k_i))
    NMLL_WITH_PRIOR = "NMLL_WITH_PRIOR"  # negative (mean) log posterior density (unnormalized) -1/B\sum_{i=1}^{B} 1/N_i [log p(y_i|X_i,\theta_{i}=g(D,k_i))+log p(\theta_{i})]
    NMLL_WITH_PRIOR_PARAM_SCALED = "NMLL_WITH_PRIOR_PARAM_SCALED"  # negative (mean) log posterior density (unnormalized) -1/B\sum_{i=1}^{B} 1/N_i 1/|\theta_{i}| [log p(y_i|X_i,\theta_{i}=g(D,k_i))+log p(\theta_{i})]
    NMLL_WITH_PRIOR_SQRT_PARAM_SCALED = "NMLL_WITH_PRIOR_SQRT_PARAM_SCALED"  # negative (mean) log posterior density (unnormalized) -1/B\sum_{i=1}^{B} 1/N_i 1/\sqrt(|\theta_{i}|) [log p(y_i|X_i,\theta_{i}=g(D,k_i))+log p(\theta_{i})]
    NMLL_WITH_NOISE_PRIOR = "NMLL_WITH_NOISE_PRIOR"  # only use noise prior instead of complete prior
    PARAMETER_RMSE = "PARAMETER_RMSE"
    PARAMETER_RMSE_PLUS_NMLL = "PARAMETER_RMSE_PLUS_NMLL"
    NOISE_RMSE_PLUS_NMLL = "NOISE_RMSE_PLUS_NMLL"


class PredictionQuantity(IntEnum):
    PREDICT_F = 0
    PREDICT_Y = 1
