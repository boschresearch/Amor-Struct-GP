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
from typing import Tuple
from amorstructgp.config.kernels.gpytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from amorstructgp.config.prior_parameters import (
    KERNEL_LENGTHSCALE_GAMMA,
    KERNEL_VARIANCE_GAMMA,
    PERIODIC_KERNEL_PERIOD_GAMMA,
    LINEAR_KERNEL_OFFSET_GAMMA,
    RQ_KERNEL_ALPHA_GAMMA,
)


class BaseElementaryKernelPytorchConfig(BaseKernelPytorchConfig):
    active_on_single_dimension: bool = False
    active_dimension: int = 0


class BasicRBFPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name = "BasicRBF"


class RBFWithPriorPytorchConfig(BasicRBFPytorchConfig):
    add_prior: bool = True
    name = "RBFwithPrior"


class BasicMatern52PytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name = "BasicMatern52"


class Matern52WithPriorPytorchConfig(BasicMatern52PytorchConfig):
    add_prior: bool = True
    name = "Matern52withPrior"


class BasicMatern32PytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    name = "BasicMatern32"


class Matern32WithPriorPytorchConfig(BasicMatern32PytorchConfig):
    add_prior: bool = True
    name = "Matern32withPrior"


class BasicPeriodicKernelPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    base_period: float = 1.0
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    period_prior_parameters: Tuple[float, float] = PERIODIC_KERNEL_PERIOD_GAMMA
    name = "BasicPeriodic"


class PeriodicWithPriorPytorchConfig(BasicPeriodicKernelPytorchConfig):
    add_prior: bool = True
    name = "PeriodicWithPrior"


class BasicRQKernelPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    base_alpha: float = 1.0
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = KERNEL_LENGTHSCALE_GAMMA
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    alpha_prior_parameters: Tuple[float, float] = RQ_KERNEL_ALPHA_GAMMA
    name = "BasicRQ"


class RQWithPriorPytorchConfig(BasicRQKernelPytorchConfig):
    add_prior: bool = True
    name = "RQwithPrior"


class BasicLinearKernelPytorchConfig(BaseElementaryKernelPytorchConfig):
    base_variance: float = 1.0
    base_offset: float = 1.0
    add_prior: bool = False
    variance_prior_parameters: Tuple[float, float] = KERNEL_VARIANCE_GAMMA
    offset_prior_parameters: Tuple[float, float] = LINEAR_KERNEL_OFFSET_GAMMA
    name = "BasicLinear"


class LinearWithPriorPytorchConfig(BasicLinearKernelPytorchConfig):
    add_prior: bool = True
    name = "LinearWithPrior"
