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
from amorstructgp.config.kernels.kernel_list_configs import BasicKernelListConfig
from amorstructgp.config.kernels.gpytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from amorstructgp.config.kernels.gpytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicLinearKernelPytorchConfig,
    BasicPeriodicKernelPytorchConfig,
    BasicRBFPytorchConfig,
    BasicRQKernelPytorchConfig,
    BasicMatern32PytorchConfig,
    BasicMatern52PytorchConfig,
)
from amorstructgp.gp.gpytorch_kernels.elementary_kernels_pytorch import (
    LinearKernelPytorch,
    PeriodicKernelPytorch,
    RBFKernelPytorch,
    RQKernelPytorch,
    Matern32KernelPytorch,
    Matern52KernelPytorch,
)


class PytorchKernelFactory:
    @staticmethod
    def build(kernel_config: BaseKernelPytorchConfig):
        if isinstance(kernel_config, BasicRBFPytorchConfig):
            return RBFKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicPeriodicKernelPytorchConfig):
            return PeriodicKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicRQKernelPytorchConfig):
            return RQKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicLinearKernelPytorchConfig):
            return LinearKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicMatern52PytorchConfig):
            return Matern52KernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicMatern32PytorchConfig):
            return Matern32KernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicKernelListConfig):
            from amorstructgp.gp.base_kernels import transform_kernel_list_to_expression

            expanded_kernel_list = [kernel_config.kernel_list for i in range(0, kernel_config.input_dimension)]
            kernel_expression = transform_kernel_list_to_expression(expanded_kernel_list, add_prior=kernel_config.add_prior)
            return kernel_expression.get_kernel()
