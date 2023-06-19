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
import gpytorch

from amorstructgp.gp.kernel_grammar import BaseKernelGrammarExpression


def get_hp_sample_from_prior_gpytorch_as_state_dict(expression: BaseKernelGrammarExpression, wrap_in_addition=True):
    kernel = expression.get_kernel()
    if wrap_in_addition:
        kernel = gpytorch.kernels.AdditiveKernel(kernel)
    kernel = kernel.pyro_sample_from_prior()
    state_dict = kernel.state_dict()
    return state_dict


def get_gpytorch_kernel_from_expression_and_state_dict(expression: BaseKernelGrammarExpression, state_dict, wrap_in_addition=True):
    kernel = expression.get_kernel()
    if wrap_in_addition:
        kernel = gpytorch.kernels.AdditiveKernel(kernel)
    kernel.load_state_dict(state_dict)
    return kernel


def print_gpytorch_parameters(module: gpytorch.Module):
    print("Model parameters:")
    for name, param, constraint in module.named_parameters_and_constraints():
        if constraint is not None:
            print(f"Parameter name: {name:55} value = {constraint.transform(param)}")
        else:
            print(f"Parameter name: {name:55} value = {param}")


def get_gpytorch_exponential_prior(lambda_param: float):
    """
    Returns an exponential prior with lambda_param as lambda -
    uses the Gamma prior of gpytorch with alpha=1 and beta=lambda
    (see for equivalence https://de.wikipedia.org/wiki/Gammaverteilung)
    """
    return gpytorch.priors.GammaPrior(1.0, lambda_param)
