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
from abc import abstractmethod
from typing import Tuple
import gpytorch
import torch
from gpytorch.constraints import Positive
from amorstructgp.gp.gpytorch_kernels.customized_gpytorch_kernels import PeriodicKernel


class BaseElementaryKernelPytorch(gpytorch.kernels.Kernel):
    def __init__(
        self,
        input_dimension: int,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        self.input_dimension = input_dimension
        self.active_dimension = active_dimension
        self.active_on_single_dimension = active_on_single_dimension

        if active_on_single_dimension:
            self.name = name + "_on_" + str(active_dimension)
            super().__init__(active_dims=torch.tensor([active_dimension]))
            self.num_active_dimensions = 1
        else:
            self.name = name
            super().__init__()
            self.num_active_dimensions = input_dimension
        self.kernel = None

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return self.kernel.forward(x1, x2, diag, last_dim_is_batch, **params)

    def get_input_dimension(self):
        return self.input_dimension

    @abstractmethod
    def get_parameters_flattened(self) -> torch.tensor:
        raise NotImplementedError


class RBFKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior)
            self.kernel = gpytorch.kernels.ScaleKernel(rbf_kernel, outputscale_prior=outputscale_prior)
        else:
            rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
        lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        rbf_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if sqrt_variance:
            variance_flattened = torch.sqrt(torch.flatten(torch.tensor([self.kernel.outputscale])))
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))
        return torch.concat((lengthscales_flattened, variance_flattened))


class Matern52KernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)
        else:
            matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel)
        lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        matern_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if sqrt_variance:
            variance_flattened = torch.sqrt(torch.flatten(torch.tensor([self.kernel.outputscale])))
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))
        return torch.concat((lengthscales_flattened, variance_flattened))


class Matern32KernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)
        else:
            matern_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel)
        lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        matern_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if sqrt_variance:
            variance_flattened = torch.sqrt(torch.flatten(torch.tensor([self.kernel.outputscale])))
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))
        return torch.concat((lengthscales_flattened, variance_flattened))


class PeriodicKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        base_period: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        period_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_period, b_period = period_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            period_prior = gpytorch.priors.GammaPrior(a_period, b_period)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            periodic_kernel = PeriodicKernel(
                ard_num_dims=self.num_active_dimensions, period_length_prior=period_prior, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(periodic_kernel, outputscale_prior=outputscale_prior)
        else:
            periodic_kernel = PeriodicKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(periodic_kernel)
        lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        periods = torch.full((1, self.num_active_dimensions), base_period)
        periodic_kernel.lengthscale = lengthscales
        periodic_kernel.period_length = periods
        self.kernel.outputscale = base_variance

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if sqrt_variance:
            variance_flattened = torch.sqrt(torch.flatten(torch.tensor([self.kernel.outputscale])))
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))
        period_flattened = torch.flatten(self.kernel.base_kernel.period_length)
        return torch.concat((lengthscales_flattened, variance_flattened, period_flattened))


class RQKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        base_alpha: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        alpha_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_alpha, b_alpha = alpha_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            alpha_prior = gpytorch.priors.GammaPrior(a_alpha, b_alpha)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            rq_kernel = gpytorch.kernels.RQKernel(ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior)
            rq_kernel.register_prior(
                "alpha_prior",
                alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m.initialize(raw_alpha=m.raw_alpha_constraint.inverse_transform(torch.to_tensor(v))),
            )
            self.kernel = gpytorch.kernels.ScaleKernel(rq_kernel, outputscale_prior=outputscale_prior)
        else:
            rq_kernel = gpytorch.kernels.RQKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(rq_kernel)
        lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        rq_kernel.lengthscale = lengthscales
        rq_kernel.alpha = torch.tensor(base_alpha)
        self.kernel.outputscale = torch.tensor(base_variance)

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if sqrt_variance:
            variance_flattened = torch.sqrt(torch.flatten(torch.tensor([self.kernel.outputscale])))
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))
        alpha_flattened = torch.flatten(torch.tensor([self.kernel.base_kernel.alpha]))
        return torch.concat((lengthscales_flattened, variance_flattened, alpha_flattened))


class LinearKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_variance: float,
        base_offset: float,
        add_prior: bool,
        variance_prior_parameters: Tuple[float, float],
        offset_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_variance, b_variance = variance_prior_parameters
        a_offset, b_offset = offset_prior_parameters
        if add_prior:
            variance_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            self.kernel = gpytorch.kernels.LinearKernel(num_dimensions=self.num_active_dimensions, variance_prior=variance_prior)
        else:
            self.kernel = gpytorch.kernels.LinearKernel(num_dimensions=self.num_active_dimensions)
        self.kernel.variance = torch.tensor(base_variance)

        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        offset_constraint = Positive()

        self.register_constraint("raw_offset", offset_constraint)

        if add_prior:
            offset_prior = gpytorch.priors.GammaPrior(a_offset, b_offset)
            self.register_prior(
                "offset_prior",
                offset_prior,
                lambda m: m.offset,
                lambda m, v: m._set_offset(v),
            )

        self.offset = base_offset

    @property
    def offset(self):
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value):
        return self._set_offset(value)

    def _set_offset(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch
        K = self.kernel.forward(x1, x2, diag, last_dim_is_batch, **params) + self.offset
        return K

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        offset_flattened = torch.flatten(torch.tensor([self.offset]))
        if sqrt_variance:
            variance_flattened = torch.sqrt(torch.flatten(torch.tensor([self.kernel.variance])))
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.variance]))
        return torch.concat((offset_flattened, variance_flattened))


if __name__ == "__main__":
    linear_kernel = LinearKernelPytorch(3, 1.0, 1.0, True, (1.0, 1.0), (1.0, 1.0), False, 0, "Linear")
    print(linear_kernel.offset)
    X = torch.randn((10, 3))
    K = linear_kernel(X)
    print(X.numpy())
    print(K.numpy())
