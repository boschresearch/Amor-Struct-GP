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

import math
from typing import Optional

import torch

from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels.kernel import Kernel


class PeriodicKernel(Kernel):
    r"""Copied and adapted from gpytorch.kernels.PeriodicKernel (https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/kernels/periodic_kernel.py - MIT Licence)
    Difference to  gpytorch.kernels.PeriodicKernel: only reparameterization - replaced scale factor -2.0 with -0.5
    and lengthscale is now squared)

    .. math::

        \begin{equation*}
            k_{\text{Periodic}}(\mathbf{x}, \mathbf{x'}) = \exp \left(
            -0.5 \sum_i
            \frac{\sin ^2 \left( \frac{\pi}{p} ({x_{i}} - {x_{i}'} ) \right)}{\lambda^2}
            \right)
        \end{equation*}

    """

    has_lengthscale = True

    def __init__(
        self,
        period_length_prior: Optional[Prior] = None,
        period_length_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(PeriodicKernel, self).__init__(**kwargs)
        if period_length_constraint is None:
            period_length_constraint = Positive()

        ard_num_dims = kwargs.get("ard_num_dims", 1)
        self.register_parameter(name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, ard_num_dims)))

        if period_length_prior is not None:
            if not isinstance(period_length_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_length_prior).__name__)
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda m: m.period_length,
                lambda m, v: m._set_period_length(v),
            )

        self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # Pop this argument so that we can manually sum over dimensions
        last_dim_is_batch = params.pop("last_dim_is_batch", False)
        # Get lengthscale
        lengthscale = self.lengthscale

        x1_ = x1.div(self.period_length / math.pi)
        x2_ = x2.div(self.period_length / math.pi)
        # We are automatically overriding last_dim_is_batch here so that we can manually sum over dimensions.
        diff = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True, **params)

        if diag:
            lengthscale = lengthscale[..., 0, :, None]
        else:
            lengthscale = lengthscale[..., 0, :, None, None]
        lengthscale_square = lengthscale.pow(2.0)
        exp_term = diff.sin().pow(2.0).div(lengthscale_square).mul(-0.5)

        if not last_dim_is_batch:
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

        return exp_term.exp()
