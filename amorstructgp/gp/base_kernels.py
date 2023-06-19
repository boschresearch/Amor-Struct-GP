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
from enum import Enum, IntEnum
from typing import List, Optional, Tuple, Union
import numpy as np
from amorstructgp.config.kernels.gpytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicLinearKernelPytorchConfig,
    BasicMatern52PytorchConfig,
    BasicPeriodicKernelPytorchConfig,
    BasicRBFPytorchConfig,
    LinearWithPriorPytorchConfig,
    Matern52WithPriorPytorchConfig,
    PeriodicWithPriorPytorchConfig,
    RBFWithPriorPytorchConfig,
)
from amorstructgp.gp.gpytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from amorstructgp.nn.dataset_encoder import get_padded_dataset_and_masks, DatasetEncoder
from amorstructgp.nn.kernel_encoder import KernelEncoder
import torch
import torch.nn as nn
import time
import math
from amorstructgp.gp.kernel_grammar import (
    BaseKernelGrammarExpression,
    BaseKernelsLibrary,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarExpressionTransformer,
    KernelGrammarOperator,
)
from amorstructgp.nn.module import MLP
from amorstructgp.utils.gpytorch_utils import get_gpytorch_kernel_from_expression_and_state_dict, print_gpytorch_parameters
from amorstructgp.config.prior_parameters import (
    KERNEL_LENGTHSCALE_GAMMA,
    KERNEL_VARIANCE_GAMMA,
    LINEAR_KERNEL_OFFSET_GAMMA,
    PERIODIC_KERNEL_PERIOD_GAMMA,
)
from amorstructgp.gp.base_symbols import BaseKernelTypes, N_BASE_KERNEL_TYPES


class BaseKernelEvalMode(Enum):
    """
    These enums determine how the Base kernels are evaluated with respect
    to their parameters (which are inputs to the forward method). The enums decide
    if embeddings should be used to determine the kernel parameters (STANDARD) or if the untransformed parameters
    are used directly (WARM_START) or if the input regarding the kernel parameters should be ignored and
    standard values should be used (DEBUG)
    """

    STANDARD = 1
    DEBUG = 2
    WARM_START = 3
    VALUE_LIST = 4


BaseKernelParameterFormat = Union[torch.tensor, List[torch.tensor]]

KernelParameterNestedList = List[List[BaseKernelParameterFormat]]

KernelTypeList = List[List[BaseKernelTypes]]


class BaseKernel:
    """
    Interface for all base kernels that are in the amortzized inference pipeline.
    A key feature in all base kernels compared e.g. to the standard kernels in the kernel folder is that the input to self.forward
    not just contains the input data X1 and X2 but also the kernel parameters (either in form of an embedding or in form of the untransformed parameter value).
    """

    @abstractmethod
    def forward(
        self,
        X1: torch.tensor,
        X2: torch.tensor,
        kernel_embedding: torch.tensor,
        untransformed_params: Optional[BaseKernelParameterFormat] = None,
    ) -> torch.tensor:
        """
        Forward pass that evaluates the kernel - has as input not just the input data X1 or X2 but also kernel_embedding (for STANDARD) and untransformed_params (for WARM_START) that deterimenes the
        kernel hyperparameter (@TODO: check if batch shapes would go through directly - if not consider extending to batch version)

        Arguments:
            X1 torch.tensor with shape N1 x D
            X2 torch.tensor with shape N2 x D
            kernel_embedding: torch.tensor of shape (kernel_embedings_dim,)
            untransformed_params: either torch.tensor with shape (self.num_params,) or list of multiple such tensors - needs to correspond to output of get_untransformed_parameters
        Returns:
            torch.tensor gram matrix with shape N1 x N2
        """
        raise NotImplementedError

    @abstractmethod
    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        """
        Method to get the log prior probability of all parameters in the kernel e.g. for the SE kernel this would return log p(lengthscale) + log p(variance)
        One can backprobagate through this method

        Arguments:
            kernel_embedding: torch.tensor of shape (kernel_embedings_dim,)
            untransformed_params: either torch.tensor with shape (self.num_params,) or list of multiple such tensors - needs to correspond to output of get_untransformed_parameters
        Returns:
            torch.tensor single value of log probability under prior of kernel parameters
        """
        raise NotImplementedError

    @abstractmethod
    def get_untransformed_parameters(self, kernel_embedding: torch.tensor) -> BaseKernelParameterFormat:
        """
        Returns the untransformed parameters that are the result of the kernel embedding e.g. the real version of the lengthscale that gets than transformed to the positive number line
        shape or list of tensors needs to correpond to the input format in the forward pass. This function is mainly used to make warm start from kernel hyperparameters that
        were set by the amortized model
        One can NOT backprobagate through this method

        Arguments:
            kernel_embedding : torch.tensor  with shape (kernel_embedings_dim,)
        Returns:
            either torch.tensor with shape (self.num_params,) or list of multiple such tensors
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self, kernel_embedding: torch.tensor, detach: bool = True) -> BaseKernelParameterFormat:
        """
        Returns the kernel parameters that are the result of the kernel embedding e.g. the lengthscale that is deterimned by the embedding.
        One can NOT backprobagate through this method

        Arguments:
            kernel_embedding : torch.tensor  with shape (kernel_embedings_dim,)
        Returns:
            either torch.tensor with shape (self.num_params,) or list of multiple such tensors
        """
        raise NotImplementedError

    def get_num_params(self) -> int:
        """
        Gives back its number of parameters

        Returns:
            int number of trainable parameters
        """
        raise NotImplementedError

    @abstractmethod
    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        raise NotImplementedError


def squared_dist(X1, X2, lengthscale, clamp_min):
    """
    Source: https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License
    """
    X1 = X1.div(lengthscale)
    X2 = X2.div(lengthscale)
    X1_norm = torch.sum(X1**2, dim=-1).unsqueeze(-1)  # B X N1 X 1
    X2_norm = torch.sum(X2**2, dim=-1).unsqueeze(-2)  # B X 1 X N2
    Distance_squared = (X1_norm + X2_norm - 2 * torch.matmul(X1, X2.transpose(-1, -2))).clamp_min_(clamp_min)
    return Distance_squared


class SEKernel(nn.Module, BaseKernel):
    def __init__(self, kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode=BaseKernelEvalMode.STANDARD) -> None:
        super().__init__()
        self.num_params = 2
        self.register_buffer("a_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[0]))
        self.register_buffer("b_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[1]))
        self.register_buffer("a_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[0]))
        self.register_buffer("b_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[1]))
        self.mlp = MLP(kernel_embedding_dim, dim_hidden_layer_list, self.num_params, dropout_p=dropout_p, use_biases=True)
        self.softplus = torch.nn.Softplus()
        self.lengthscale_index = 0
        self.variance_index = 1
        self.eval_mode = eval_mode
        self.debug_lengthscale = torch.tensor(1.0)
        self.debug_variance = torch.tensor(1.0)

    def get_lengthscale_prior(self):
        return torch.distributions.Gamma(self.a_lengthscale, self.b_lengthscale)

    def get_variance_prior(self):
        return torch.distributions.Gamma(self.a_variance, self.b_variance)

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode

    def forward(self, X1, X2, kernel_embedding, untransformed_params=None):
        # x1: N1 x D
        lengthscale, variance = self._get_parameters(kernel_embedding, untransformed_params)
        sq_distances = squared_dist(X1, X2, lengthscale, 0)
        K = torch.pow(variance, 2.0) * torch.exp(-0.5 * sq_distances)
        return K

    def _get_parameters(self, kernel_embedding, untransformed_params):
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            lengthscale = self.debug_lengthscale
            variance = self.debug_variance
        elif self.eval_mode == BaseKernelEvalMode.STANDARD:
            params = self.softplus(self.mlp(kernel_embedding))
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.WARM_START:
            params = self.softplus(untransformed_params)
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.VALUE_LIST:
            params = untransformed_params
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
        return lengthscale, variance

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        lengthscale, variance = self._get_parameters(kernel_embedding, untransformed_params)
        return self._get_log_prior_value(lengthscale, variance)

    def _get_log_prior_value(self, lengthscale, variance):
        log_p_lengthscale = self.get_lengthscale_prior().log_prob(lengthscale)
        log_p_variance = self.get_variance_prior().log_prob(variance)
        return log_p_lengthscale + log_p_variance

    def get_untransformed_parameters(self, kernel_embedding):
        return self.mlp(kernel_embedding).clone().detach().requires_grad_(True)

    def get_parameters(self, kernel_embedding, detach: bool = True):
        if detach:
            return self.softplus(self.mlp(kernel_embedding)).clone().detach().requires_grad_(True)
        else:
            return self.softplus(self.mlp(kernel_embedding))

    def get_num_params(self) -> int:
        return self.num_params


class MaternKernel(nn.Module, BaseKernel):
    def __init__(self, kernel_embedding_dim, dim_hidden_layer_list, dropout_p, nu: float, eval_mode=BaseKernelEvalMode.STANDARD) -> None:
        super().__init__()
        assert nu == 1.5 or nu == 2.5
        self.nu = nu
        self.num_params = 2
        self.register_buffer("a_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[0]))
        self.register_buffer("b_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[1]))
        self.register_buffer("a_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[0]))
        self.register_buffer("b_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[1]))
        self.mlp = MLP(kernel_embedding_dim, dim_hidden_layer_list, self.num_params, dropout_p=dropout_p, use_biases=True)
        self.softplus = torch.nn.Softplus()
        self.lengthscale_index = 0
        self.variance_index = 1
        self.eval_mode = eval_mode
        self.debug_lengthscale = torch.tensor(1.0)
        self.debug_variance = torch.tensor(1.0)

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode

    def get_lengthscale_prior(self):
        return torch.distributions.Gamma(self.a_lengthscale, self.b_lengthscale)

    def get_variance_prior(self):
        return torch.distributions.Gamma(self.a_variance, self.b_variance)

    def forward(self, X1, X2, kernel_embedding, untransformed_params=None):
        # x1: N1 x D
        lengthscale, variance = self._get_parameters(kernel_embedding, untransformed_params)
        sq_distances = squared_dist(X1, X2, lengthscale, 1e-6)
        distance = torch.sqrt(sq_distances)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
        if self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * (distance) ** 2)
        K = torch.pow(variance, 2.0) * torch.mul(constant_component, exp_component)
        return K

    def _get_parameters(self, kernel_embedding, untransformed_params):
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            lengthscale = self.debug_lengthscale
            variance = self.debug_variance
        elif self.eval_mode == BaseKernelEvalMode.STANDARD:
            params = self.softplus(self.mlp(kernel_embedding))
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.WARM_START:
            params = self.softplus(untransformed_params)
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.VALUE_LIST:
            params = untransformed_params
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
        return lengthscale, variance

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        lengthscale, variance = self._get_parameters(kernel_embedding, untransformed_params)
        return self._get_log_prior_value(lengthscale, variance)

    def _get_log_prior_value(self, lengthscale, variance):
        log_p_lengthscale = self.get_lengthscale_prior().log_prob(lengthscale)
        log_p_variance = self.get_variance_prior().log_prob(variance)
        return log_p_lengthscale + log_p_variance

    def get_untransformed_parameters(self, kernel_embedding):
        return self.mlp(kernel_embedding).clone().detach().requires_grad_(True)

    def get_parameters(self, kernel_embedding, detach: bool = True):
        if detach:
            return self.softplus(self.mlp(kernel_embedding)).clone().detach().requires_grad_(True)
        else:
            return self.softplus(self.mlp(kernel_embedding))

    def get_num_params(self) -> int:
        return self.num_params


class PeriodicKernel(nn.Module, BaseKernel):
    def __init__(self, kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode=BaseKernelEvalMode.STANDARD) -> None:
        super().__init__()
        self.num_params = 3
        self.mlp = MLP(kernel_embedding_dim, dim_hidden_layer_list, self.num_params, dropout_p=dropout_p, use_biases=True)
        self.softplus = torch.nn.Softplus()
        self.register_buffer("a_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[0]))
        self.register_buffer("b_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[1]))
        self.register_buffer("a_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[0]))
        self.register_buffer("b_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[1]))
        self.register_buffer("a_period", torch.tensor(PERIODIC_KERNEL_PERIOD_GAMMA[0]))
        self.register_buffer("b_period", torch.tensor(PERIODIC_KERNEL_PERIOD_GAMMA[1]))
        self.lengthscale_index = 0
        self.variance_index = 1
        self.period_index = 2
        self.eval_mode = eval_mode
        self.debug_lengthscale = torch.tensor(1.0)
        self.debug_variance = torch.tensor(1.0)
        self.debug_period = torch.tensor(1.0)

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode

    def get_lengthscale_prior(self):
        return torch.distributions.Gamma(self.a_lengthscale, self.b_lengthscale)

    def get_variance_prior(self):
        return torch.distributions.Gamma(self.a_variance, self.b_variance)

    def get_period_prior(self):
        return torch.distributions.Gamma(self.a_period, self.b_period)

    def forward(self, X1, X2, kernel_embedding, untransformed_params=None):
        # x1: N1 x D
        # x2: N2 x D
        lengthscale, variance, period = self._get_parameters(kernel_embedding, untransformed_params)
        X1 = X1.div(period).unsqueeze(-2)  # N X 1 X D
        X2 = X2.div(period).unsqueeze(-3)  # 1 x N x D
        X_diff = torch.abs(X1 - X2)  # N x N x D
        K = (-0.5 * (torch.sin(math.pi * X_diff) ** 2) / lengthscale**2).exp_()  # N1 X N2 X D
        K = torch.pow(variance, 2.0) * torch.prod(K, -1)
        return K

    def _get_parameters(self, kernel_embedding, untransformed_params):
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            lengthscale = self.debug_lengthscale
            variance = self.debug_variance
            period = self.debug_period
        elif self.eval_mode == BaseKernelEvalMode.STANDARD:
            params = self.softplus(self.mlp(kernel_embedding))
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
            period = params[self.period_index]
        elif self.eval_mode == BaseKernelEvalMode.WARM_START:
            params = self.softplus(untransformed_params)
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
            period = params[self.period_index]
        elif self.eval_mode == BaseKernelEvalMode.VALUE_LIST:
            params = untransformed_params
            lengthscale = params[self.lengthscale_index]
            variance = params[self.variance_index]
            period = params[self.period_index]
        return lengthscale, variance, period

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        lengthscale, variance, period = self._get_parameters(kernel_embedding, untransformed_params)
        return self._get_log_prior_value(lengthscale, variance, period)

    def _get_log_prior_value(self, lengthscale, variance, period):
        log_p_lengthscale = self.get_lengthscale_prior().log_prob(lengthscale)
        log_p_variance = self.get_variance_prior().log_prob(variance)
        log_p_period = self.get_period_prior().log_prob(period)
        return log_p_lengthscale + log_p_variance + log_p_period

    def get_untransformed_parameters(self, kernel_embedding):
        return self.mlp(kernel_embedding).clone().detach().requires_grad_(True)

    def get_parameters(self, kernel_embedding, detach: bool = True):
        if detach:
            return self.softplus(self.mlp(kernel_embedding)).clone().detach().requires_grad_(True)
        else:
            return self.softplus(self.mlp(kernel_embedding))

    def get_num_params(self) -> int:
        return self.num_params


class LinearKernel(nn.Module, BaseKernel):
    def __init__(self, kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode=BaseKernelEvalMode.STANDARD) -> None:
        super().__init__()
        self.num_params = 2
        self.mlp = MLP(kernel_embedding_dim, dim_hidden_layer_list, self.num_params, dropout_p=dropout_p, use_biases=True)
        self.softplus = torch.nn.Softplus()
        self.register_buffer("a_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[0]))
        self.register_buffer("b_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[1]))
        self.register_buffer("a_offset", torch.tensor(LINEAR_KERNEL_OFFSET_GAMMA[0]))
        self.register_buffer("b_offset", torch.tensor(LINEAR_KERNEL_OFFSET_GAMMA[1]))
        self.offset_index = 0
        self.variance_index = 1
        self.eval_mode = eval_mode
        self.debug_offset = torch.tensor(1.0)
        self.debug_variance = torch.tensor(1.0)

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode

    def get_variance_prior(self):
        return torch.distributions.Gamma(self.a_variance, self.b_variance)

    def get_offset_prior(self):
        return torch.distributions.Gamma(self.a_offset, self.b_offset)

    def forward(self, X1, X2, kernel_embedding, untransformed_params=None):
        # x1: N1 x D
        # x2: N2 x D
        offset, variance = self._get_parameters(kernel_embedding, untransformed_params)
        K = torch.pow(variance, 2.0) * torch.matmul(X1, X2.transpose(-1, -2)) + offset
        return K

    def _get_parameters(self, kernel_embedding, untransformed_params):
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            offset = self.debug_offset
            variance = self.debug_variance
        elif self.eval_mode == BaseKernelEvalMode.STANDARD:
            params = self.softplus(self.mlp(kernel_embedding))
            offset = params[self.offset_index]
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.WARM_START:
            params = self.softplus(untransformed_params)
            offset = params[self.offset_index]
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.VALUE_LIST:
            params = untransformed_params
            offset = params[self.offset_index]
            variance = params[self.variance_index]
        return offset, variance

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        offset, variance = self._get_parameters(kernel_embedding, untransformed_params)
        return self._get_log_prior_value(offset, variance)

    def _get_log_prior_value(self, offset, variance):
        log_p_offset = self.get_offset_prior().log_prob(offset)
        log_p_variance = self.get_variance_prior().log_prob(variance)
        return log_p_offset + log_p_variance

    def get_untransformed_parameters(self, kernel_embedding):
        return self.mlp(kernel_embedding).clone().detach().requires_grad_(True)

    def get_parameters(self, kernel_embedding, detach: bool = True):
        if detach:
            return self.softplus(self.mlp(kernel_embedding)).clone().detach().requires_grad_(True)
        else:
            return self.softplus(self.mlp(kernel_embedding))

    def get_num_params(self) -> int:
        return self.num_params


class KernelAddition(nn.Module, BaseKernel):
    def __init__(self, kernel1: BaseKernel, kernel2: BaseKernel):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.eval_param_list = lambda params, i: None if params is None else params[i]

    def forward(self, x1, x2, kernel_embedding, untransformed_params=None):
        K = self.kernel1.forward(x1, x2, kernel_embedding, self.eval_param_list(untransformed_params, 0)) + self.kernel2.forward(
            x1, x2, kernel_embedding, self.eval_param_list(untransformed_params, 1)
        )
        return K

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        log_prior_prob_k1 = self.kernel1.get_log_prior_prob(kernel_embedding, self.eval_param_list(untransformed_params, 0))
        log_prior_prob_k2 = self.kernel2.get_log_prior_prob(kernel_embedding, self.eval_param_list(untransformed_params, 1))
        return log_prior_prob_k1 + log_prior_prob_k2

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.kernel1.set_eval_mode(eval_mode)
        self.kernel2.set_eval_mode(eval_mode)

    def get_untransformed_parameters(self, kernel_embedding):
        return [self.kernel1.get_untransformed_parameters(kernel_embedding), self.kernel2.get_untransformed_parameters(kernel_embedding)]

    def get_parameters(self, kernel_embedding, detach: bool = True):
        return [self.kernel1.get_parameters(kernel_embedding, detach), self.kernel2.get_parameters(kernel_embedding, detach)]

    def get_num_params(self) -> int:
        return self.kernel1.get_num_params() + self.kernel2.get_num_params()


class KernelMultiplication(nn.Module, BaseKernel):
    def __init__(self, kernel1: BaseKernel, kernel2: BaseKernel):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.eval_param_list = lambda params, i: None if params is None else params[i]

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.kernel1.set_eval_mode(eval_mode)
        self.kernel2.set_eval_mode(eval_mode)

    def forward(self, x1, x2, kernel_embedding, untransformed_params=None):
        K = self.kernel1.forward(x1, x2, kernel_embedding, self.eval_param_list(untransformed_params, 0)) * self.kernel2.forward(
            x1, x2, kernel_embedding, self.eval_param_list(untransformed_params, 1)
        )
        return K

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        log_prior_prob_k1 = self.kernel1.get_log_prior_prob(kernel_embedding, self.eval_param_list(untransformed_params, 0))
        log_prior_prob_k2 = self.kernel2.get_log_prior_prob(kernel_embedding, self.eval_param_list(untransformed_params, 1))
        return log_prior_prob_k1 + log_prior_prob_k2

    def get_untransformed_parameters(self, kernel_embedding):
        return [self.kernel1.get_untransformed_parameters(kernel_embedding), self.kernel2.get_untransformed_parameters(kernel_embedding)]

    def get_parameters(self, kernel_embedding, detach: bool = True):
        return [self.kernel1.get_parameters(kernel_embedding, detach), self.kernel2.get_parameters(kernel_embedding, detach)]

    def get_num_params(self) -> int:
        return self.kernel1.get_num_params() + self.kernel2.get_num_params()


class DimWiseAdditiveKernelWrapper(nn.Module):
    """
    Module that can calculate the gram matrix/ evaluate kernels that are additions of BaseKernels inside each dimension and multiplications between dimensions.
    """

    def __init__(
        self,
        kernel_embedding_dim: int,
        dim_hidden_layer_list: List[int],
        dropout_p: float,
        share_weights_in_additions: bool,
        eval_mode=BaseKernelEvalMode.STANDARD,
    ):
        super().__init__()
        self.eval_mode = eval_mode
        share_weights_in_additions = share_weights_in_additions
        se_kernel = SEKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode)
        lin_kernel = LinearKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode)
        per_kernel = PeriodicKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode)
        matern52_kernel = MaternKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, 2.5, eval_mode)

        self.eval_param_list = lambda params, d, i: None if params is None else params[d][i]

        if share_weights_in_additions:
            # If this is active we only have linear feature extractors for SE, PER and LIN -> they are also used to extract the hps from the embeddings in the multiplicative terms
            se_mult_lin = KernelMultiplication(se_kernel, lin_kernel)
            se_mult_per = KernelMultiplication(se_kernel, per_kernel)
            se_mult_matern52 = KernelMultiplication(se_kernel, matern52_kernel)
            lin_mult_per = KernelMultiplication(lin_kernel, per_kernel)
            lin_mult_matern52 = KernelMultiplication(lin_kernel, matern52_kernel)
            per_mult_matern52 = KernelMultiplication(per_kernel, matern52_kernel)

        else:
            se_mult_lin = KernelMultiplication(
                SEKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
                LinearKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
            )
            se_mult_per = KernelMultiplication(
                SEKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
                PeriodicKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
            )
            se_mult_matern52 = KernelMultiplication(
                SEKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
                MaternKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, 2.5, eval_mode),
            )
            lin_mult_per = KernelMultiplication(
                LinearKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
                PeriodicKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
            )
            lin_mult_matern52 = KernelMultiplication(
                LinearKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
                MaternKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, 2.5, eval_mode),
            )
            per_mult_matern52 = KernelMultiplication(
                PeriodicKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode),
                MaternKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, 2.5, eval_mode),
            )

        self.kernel_module_list = nn.ModuleList(
            [
                se_kernel,
                lin_kernel,
                per_kernel,
                matern52_kernel,
                se_mult_lin,
                se_mult_per,
                se_mult_matern52,
                lin_mult_per,
                lin_mult_matern52,
                per_mult_matern52,
            ]
        )  # SE = 0 LIN = 1 PER = 2 MATERN52 = 3 SE_MULT_LIN = 4 SE_MULT_PER = 5 SE_MULT_MATERN52 = 6 LIN_MULT_PER = 7 LIN_MULT_MATERN52 = 8 PER_MULT_MATERN52 = 9

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode
        for kernel_module in self.kernel_module_list:
            assert isinstance(kernel_module, BaseKernel)
            kernel_module.set_eval_mode(eval_mode)

    def forward(
        self,
        X1: torch.tensor,
        X2: torch.tensor,
        kernel_embeddings: torch.tensor,
        kernel_type_list: KernelTypeList,
        untransformed_params: Optional[KernelParameterNestedList] = None,
    ):
        """
        The main forward method gets besides the input data X1 and X2 as input the kernel description in form of a nested list kernel_type_list (KernelTypeList): List[List[BaseKernelTypes]]
        where the first list is over dimension and the second specifies the base kernels inside each dimension (combined via addition in each dimension and via multiplication between dimensions)
        e.g. [[BaseKernelTypes.SE,BaseKernelTypes.LIN],[BaseKernelTypes.LIN,BaseKernelTypes.PER]] describes the kernel (SE_0 + LIN_0) x (LIN_1 + PER_1).
        The kernel parameters are also an input given as kernel_embeddings tensor with shape D x N_K x d_k where kernel_embeddings[d,i,:] is
        the embedding for the i-th base kernel in the d-th dimension (differences between different length kernel lists are masked in the tensor)

        Arguments:
            X1: torch.tensor with shape N1 x D
            X2: torch.tensor with shape N2 x D
            kernel_embeddings: torch.tensor with shape D x N_k x d_k
            kernel_type_list: nested list with BaseKernelTypes elements specifying the kernel structure where the first list is over dimension and the second specifies the base kernels inside each dimension
            untransformed_params: nested list that mirros the nested structure of kernel_type_list but each element contains BaseKernelParameterFormat elements - with this the
                kernel can be evaluated with parameters directly rather than embeddings (can be used for warm start learning for example)
        """

        kernel_grams = []
        # iterate over dimension
        for d, kernel_list_per_dim in enumerate(kernel_type_list):
            # iterate in each dimension over respective kernel list
            kernel_grams_per_dim = [
                self.kernel_module_list[int(kernel_type)].forward(
                    X1[:, d].unsqueeze(-1),
                    X2[:, d].unsqueeze(-1),
                    kernel_embeddings[d, i, :],
                    self.eval_param_list(untransformed_params, d, i),
                )
                for i, kernel_type in enumerate(kernel_list_per_dim)
            ]
            # sum the gram matrices inside each dimension
            kernel_gram_at_d = torch.sum(torch.stack(kernel_grams_per_dim), dim=0)
            kernel_grams.append(kernel_gram_at_d)
        # multiply the gram matrices of all dimensions
        K = torch.prod(torch.stack(kernel_grams), dim=0)  # N1 x N2
        return K

    def get_log_prior_prob(
        self,
        kernel_embeddings: torch.tensor,
        kernel_type_list: KernelTypeList,
        untransformed_params: Optional[KernelParameterNestedList] = None,
    ) -> torch.tensor:
        """
        Method to get the log prior prob of the kernel parameters that are given when evaluating the kernel given via kernel_type_list with the kernel_embeddings

        Arguments:
            kernel_embeddings: torch.tensor with shape D x N_k x d_k
            kernel_type_list: nested list with BaseKernelTypes elements specifying the kernel structure where the first list is over dimension and the second specifies the base kernels inside each dimension
            untransformed_params: nested list that mirros the nested structure of kernel_type_list but each element contains BaseKernelParameterFormat elements - with this the
                kernel can be evaluated with parameters directly rather than embeddings (can be used for warm start learning for example)
        """
        log_prior_probs = []
        # iterate over dimension
        for d, kernel_list_per_dim in enumerate(kernel_type_list):
            # iterate in each dimension over respective kernel list
            log_prior_probs_per_dim = [
                self.kernel_module_list[int(kernel_type)].get_log_prior_prob(
                    kernel_embeddings[d, i, :],
                    self.eval_param_list(untransformed_params, d, i),
                )
                for i, kernel_type in enumerate(kernel_list_per_dim)
            ]
            # sum the log prior probs inside each dimension
            log_prior_probs_at_d = torch.sum(torch.stack(log_prior_probs_per_dim), dim=0)
            log_prior_probs.append(log_prior_probs_at_d)
        # sum the log prior probs of all dimensions
        log_prior_prob = torch.sum(torch.stack(log_prior_probs), dim=0)
        return log_prior_prob

    def get_num_params(self, kernel_type_list: KernelTypeList) -> int:
        num_params = 0
        for kernel_list_per_dim in kernel_type_list:
            for kernel_type in kernel_list_per_dim:
                num_params += self.kernel_module_list[int(kernel_type)].get_num_params()
        return num_params

    def get_untransformed_parameters(self, kernel_embeddings: torch.tensor, kernel_type_list: KernelTypeList):
        untransformed_kernel_parms = []
        for d, kernel_list_per_dim in enumerate(kernel_type_list):
            untransformed_kernel_parms_per_dim = [
                self.kernel_module_list[int(kernel_type)].get_untransformed_parameters(kernel_embeddings[d, i, :])
                for i, kernel_type in enumerate(kernel_list_per_dim)
            ]
            untransformed_kernel_parms.append(untransformed_kernel_parms_per_dim)
        return untransformed_kernel_parms

    def get_parameters(self, kernel_embeddings: torch.tensor, kernel_type_list: KernelTypeList, detach: bool = True):
        kernel_parms = []
        for d, kernel_list_per_dim in enumerate(kernel_type_list):
            kernel_parms_per_dim = [
                self.kernel_module_list[int(kernel_type)].get_parameters(kernel_embeddings[d, i, :], detach)
                for i, kernel_type in enumerate(kernel_list_per_dim)
            ]
            kernel_parms.append(kernel_parms_per_dim)
        return kernel_parms


class BatchedDimWiseAdditiveKernelWrapper(nn.Module):
    """
    Batch version of DimWiseAdditiveKernelWrapper
    """

    def __init__(
        self,
        kernel_embedding_dim: int,
        dim_hidden_layer_list: List[int],
        dropout_p: float,
        share_weights_in_additions: bool,
        eval_mode=BaseKernelEvalMode.STANDARD,
    ):
        super().__init__()
        self.eval_mode = eval_mode
        self.dim_wise_additive_kernel_wrapper = DimWiseAdditiveKernelWrapper(
            kernel_embedding_dim, dim_hidden_layer_list, dropout_p, share_weights_in_additions, eval_mode
        )
        self.eval_param_list = lambda params, i: None if params is None else params[i]

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode
        self.dim_wise_additive_kernel_wrapper.set_eval_mode(eval_mode)

    def forward(
        self,
        X1: torch.tensor,
        X2: torch.tensor,
        kernel_embeddings: torch.tensor,
        kernel_type_list: List[KernelTypeList],
        untransformed_params: Optional[List[KernelParameterNestedList]] = None,
    ):
        # kernel_embeddings: B x D x N_k x d_h
        # X1: B x N1 x D
        # X2: B x N2 x D
        batch_size = X1.size(0)
        K_batch = torch.stack(
            [
                self.dim_wise_additive_kernel_wrapper.forward(
                    X1[b], X2[b], kernel_embeddings[b], kernel_type_list[b], self.eval_param_list(untransformed_params, b)
                )
                for b in range(batch_size)
            ]
        )
        return K_batch  # B X N1 X N2

    def get_log_prior_prob(
        self,
        kernel_embeddings: torch.tensor,
        kernel_type_list: List[KernelTypeList],
        untransformed_params: Optional[List[KernelParameterNestedList]] = None,
    ) -> torch.tensor:
        # kernel_embeddings: B x D x N_k x d_h
        batch_size = kernel_embeddings.size(0)
        log_prior_prob_batch = torch.stack(
            [
                self.dim_wise_additive_kernel_wrapper.get_log_prior_prob(
                    kernel_embeddings[b], kernel_type_list[b], self.eval_param_list(untransformed_params, b)
                )
                for b in range(batch_size)
            ]
        )
        return log_prior_prob_batch  # B

    def get_num_params(self, kernel_type_list: List[KernelTypeList]) -> List[int]:
        num_params = [self.dim_wise_additive_kernel_wrapper.get_num_params(kernel_list) for kernel_list in kernel_type_list]
        return num_params

    def get_untransformed_parameters(self, kernel_embeddings: torch.tensor, kernel_type_list: List[KernelTypeList]):
        batch_size = kernel_embeddings.size(0)
        untransformed_params_over_batch = [
            self.dim_wise_additive_kernel_wrapper.get_untransformed_parameters(kernel_embeddings[b], kernel_type_list[b])
            for b in range(batch_size)
        ]
        return untransformed_params_over_batch

    def get_parameters(self, kernel_embeddings: torch.tensor, kernel_type_list: List[KernelTypeList], detach: bool = True):
        batch_size = kernel_embeddings.size(0)
        params_over_batch = [
            self.dim_wise_additive_kernel_wrapper.get_parameters(kernel_embeddings[b], kernel_type_list[b], detach)
            for b in range(batch_size)
        ]
        return params_over_batch


def create_kernel_embedding_tensor(kernel_list: List[BaseKernelTypes]) -> np.array:
    """
    Creates for a single kernel list e.g [SE,LIN_PLUS_PER,PER] an one-hot embedding tensor

    Arguments:
        kernel_list: list of BaseKernelTypes - usually describes the base kernels used in a addition insider one dimension

    Returns:
        np.array: one-hot embedding array with shape LenList X EmbeddingDim

    """

    def assign(array, index, value):
        array[index] = value
        return array

    # create one-hot embedding vectors for each kernel aka SE=(1,0,0,0,0,0)... and stack them
    embedding_tensor = np.stack([assign(np.zeros(N_BASE_KERNEL_TYPES, dtype=np.float32), kernel_type, 1.0) for kernel_type in kernel_list])
    return embedding_tensor


def get_kernel_embeddings_and_kernel_mask(kernel_list: List[KernelTypeList]) -> Tuple[np.array, np.array]:
    """
    Returns kernel embding tensor and kernel mask - compatible to the input of KernelEncoder

    Arguments:
        kernel_list : first list is over batches, second over dimensions, third pver kernel types in this dimension

    Return:
        np.array - kernel embedding tensor with shape B x MaxDIM x MaxLenKernelTypes x EmbeddingDim aka B X D X N_k x d_k
        np.array - mask with shape B x MaxDim x MaxLenKernelTypes
    """
    B = len(kernel_list)
    # assert isinstance(kernel_list[0][0][0],BaseKernelTypes)
    dims = [len(kernel_list_over_dims) for kernel_list_over_dims in kernel_list]
    # Deduce what is maximum dimension in batch/kernel_list and what is the max number of base kernels present in one dimension
    max_dim = max(dims)
    max_len = max(
        [
            max([len(kernel_list_over_single_dim) for kernel_list_over_single_dim in kernel_list_over_dims])
            for kernel_list_over_dims in kernel_list
        ]
    )
    kernel_embedding_list = []
    mask_list = []
    for kernel_list_over_dims in kernel_list:
        kernel_embeddings_over_dims_list = []
        masks_over_dims_list = []
        for kernel_list_over_single_dim in kernel_list_over_dims:
            kernel_list_length = len(kernel_list_over_single_dim)
            # Create embedding tensor over kernel list for example for [SE,LIN,LIN_PLUS_PER]
            embedding_tensor = create_kernel_embedding_tensor(kernel_list_over_single_dim)
            # Calculate how much needs to be padded to the biggest kernel list
            padding_size = max_len - kernel_list_length
            padded_embedding_tensor = np.pad(embedding_tensor, ((0, padding_size), (0, 0)))  # MaxLen x EmbeddingDim
            mask_single_dim = np.pad(np.ones(kernel_list_length), ((0, padding_size)))  # MaxLen
            kernel_embeddings_over_dims_list.append(padded_embedding_tensor)
            masks_over_dims_list.append(mask_single_dim)
        kernel_embeddings_over_dims = np.stack(kernel_embeddings_over_dims_list)  # D x MaxLen x EmbeddingDim
        mask_over_dims = np.stack(masks_over_dims_list)  # D X MaxLen
        dim = len(kernel_embeddings_over_dims_list)  # D
        dim_padding_size = max_dim - dim
        kernel_embeddings_over_dims = np.pad(
            kernel_embeddings_over_dims, ((0, dim_padding_size), (0, 0), (0, 0))
        )  # MaxDim x MaxLen x EmbeddingDim
        mask_over_dims = np.pad(mask_over_dims, ((0, dim_padding_size), (0, 0)))  # MaxDim X MaxLen
        kernel_embedding_list.append(kernel_embeddings_over_dims)
        mask_list.append(mask_over_dims)

    kernel_embeddings = np.stack(kernel_embedding_list)
    mask = np.stack(mask_list)
    return kernel_embeddings, mask


def transform_kernel_list_to_expression(kernel_list: KernelTypeList, add_prior=True, make_deep_copy=True) -> BaseKernelGrammarExpression:
    """
    Transforms a kernel_list to an BaseKernelGrammarExpression
    The outer list is over dimensions, the inner list over kernel symbols in that dimension
    The final kernel is a multiplication over the dimensions, and an addition of the base kernels within the dimension
    """
    final_expression, _ = get_expressions_from_kernel_list(kernel_list, add_prior, make_deep_copy)
    return final_expression


def get_expressions_from_kernel_list(kernel_list, add_prior=True, make_deep_copy=True):
    num_dimension = len(kernel_list)
    kernel_expression_list = []
    single_expression_list = []
    for d in range(0, num_dimension):
        kernel_expression_list_dimension = []
        for kernel_type in kernel_list[d]:
            expression = get_kernel_expression_for_base_kernel_type(kernel_type, num_dimension, d, add_prior)
            kernel_expression_list_dimension.append(expression)
        single_expression_list.append(kernel_expression_list_dimension)
        expression_for_dimension = KernelGrammarExpressionTransformer.add_expressions(
            kernel_expression_list_dimension, make_deep_copy=make_deep_copy
        )
        kernel_expression_list.append(expression_for_dimension)
    final_expression = KernelGrammarExpressionTransformer.multiply_expressions(kernel_expression_list, make_deep_copy=make_deep_copy)
    return final_expression, single_expression_list


def get_parameter_value_lists(kernel_list: KernelTypeList, kernel_state_dict, wrap_in_addition: bool = True) -> KernelParameterNestedList:
    """
    Method to create a list of the parameters in kernel_state_dict that maps to the base kernels in kernel_list
    kernel_state_dict needs to be assiocated with a expression that comes from the method get_expressions_from_kernel_list
    of the respective kernel list.
    """
    # create kernel expression - non deep copy
    kernel_expression, single_expression_list = get_expressions_from_kernel_list(kernel_list, add_prior=True, make_deep_copy=False)
    # initialize the kernel from the kernel expression with the state dict - the base kernels in the elementary expressions will also have these paramters than
    get_gpytorch_kernel_from_expression_and_state_dict(kernel_expression, kernel_state_dict, wrap_in_addition=wrap_in_addition)
    # create a list of the parameters in the same format (same nested order) than the kernel_list
    value_list = get_parameter_value_list_from_expression_list(single_expression_list)
    return value_list


def get_parameter_value_list_from_expression_list(
    single_expression_list: List[List[BaseKernelGrammarExpression]],
) -> KernelParameterNestedList:
    """
    Extracts from a nested list of expressions (must be either elementary expressions of an expression with only one operator)
    the parameters of the base kernels in each expression. Returns the parameters as the same structured nested list
    """
    value_list = []
    for dim_expression_list in single_expression_list:
        dim_value_list = []
        for expression in dim_expression_list:
            assert expression.get_base_kernel_library() == BaseKernelsLibrary.GPYTORCH
            if isinstance(expression, ElementaryKernelGrammarExpression):
                params = expression.get_kernel().get_parameters_flattened().detach()
            elif isinstance(expression, KernelGrammarExpression):
                assert isinstance(expression.get_left_expression(), ElementaryKernelGrammarExpression)
                assert isinstance(expression.get_right_expression(), ElementaryKernelGrammarExpression)
                left_kernel = expression.get_left_expression().get_kernel()
                params_left = left_kernel.get_parameters_flattened().detach()
                right_kernel = expression.get_right_expression().get_kernel()
                params_right = right_kernel.get_parameters_flattened().detach()
                params = [params_left, params_right]
            else:
                raise ValueError
            dim_value_list.append(params)
        value_list.append(dim_value_list)
    return value_list


def get_kernel_expression_for_base_kernel_type(
    base_kernel_type: BaseKernelTypes, input_dimension: int, active_dimension: int, with_prior: bool = True
) -> BaseKernelGrammarExpression:
    if with_prior:
        se_config = RBFWithPriorPytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
        lin_config = LinearWithPriorPytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
        per_config = PeriodicWithPriorPytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
        matern52_config = Matern52WithPriorPytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
    else:
        se_config = BasicRBFPytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
        lin_config = BasicLinearKernelPytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
        per_config = BasicPeriodicKernelPytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
        matern52_config = BasicMatern52PytorchConfig(
            input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=active_dimension
        )
    if base_kernel_type == BaseKernelTypes.SE:
        expression = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(se_config))
    elif base_kernel_type == BaseKernelTypes.LIN:
        expression = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(lin_config))
    elif base_kernel_type == BaseKernelTypes.PER:
        expression = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(per_config))
    elif base_kernel_type == BaseKernelTypes.MATERN52:
        expression = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(matern52_config))
    elif base_kernel_type == BaseKernelTypes.LIN_MULT_PER:
        expression1 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(lin_config))
        expression2 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(per_config))
        expression = KernelGrammarExpression(expression1, expression2, KernelGrammarOperator.MULTIPLY)
    elif base_kernel_type == BaseKernelTypes.SE_MULT_LIN:
        expression1 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(se_config))
        expression2 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(lin_config))
        expression = KernelGrammarExpression(expression1, expression2, KernelGrammarOperator.MULTIPLY)
    elif base_kernel_type == BaseKernelTypes.SE_MULT_PER:
        expression1 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(se_config))
        expression2 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(per_config))
        expression = KernelGrammarExpression(expression1, expression2, KernelGrammarOperator.MULTIPLY)
    elif base_kernel_type == BaseKernelTypes.SE_MULT_MATERN52:
        expression1 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(se_config))
        expression2 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(matern52_config))
        expression = KernelGrammarExpression(expression1, expression2, KernelGrammarOperator.MULTIPLY)
    elif base_kernel_type == BaseKernelTypes.LIN_MULT_MATERN52:
        expression1 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(lin_config))
        expression2 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(matern52_config))
        expression = KernelGrammarExpression(expression1, expression2, KernelGrammarOperator.MULTIPLY)
    elif base_kernel_type == BaseKernelTypes.PER_MULT_MATERN52:
        expression1 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(per_config))
        expression2 = ElementaryKernelGrammarExpression(PytorchKernelFactory.build(matern52_config))
        expression = KernelGrammarExpression(expression1, expression2, KernelGrammarOperator.MULTIPLY)
    else:
        raise NotImplementedError
    return expression


def get_batch_from_nested_parameter_list(
    kernel_parameter_lists: List[KernelParameterNestedList], device=torch.device("cpu")
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Transforms a batch of KernelParameterNestedList objects to a torch.tensor. It first flattens all the  KernelParameterNestedList object
    padds them to the same size via 0 padding and stacks them to a batch of size B x N_max where N_max is the maximum length of the flattened
    KernelParameterNestedList object. It also returns the size of the flattened tensors

    Returns:
        torch.tensor of size B x N_max - batchified parameter lists
        torch.tensor of size B x 1 - tensor containing the lengths of the flattened tensors
    """
    flattened_tensor_list = [torch.concat(flatten_nested_list(kernel_parameter_list)) for kernel_parameter_list in kernel_parameter_lists]
    flattened_tensor_lengths = torch.tensor([len(flattened_tensor) for flattened_tensor in flattened_tensor_list])
    max_length = max(flattened_tensor_lengths)
    batched_parameter_lists = torch.stack(
        [
            torch.concat((flattened_tensor, torch.zeros(max_length - flattened_tensor_lengths[i], device=device)))
            for i, flattened_tensor in enumerate(flattened_tensor_list)
        ]
    )
    return batched_parameter_lists, flattened_tensor_lengths


def flatten_nested_list(liste):
    if len(liste) == 0:
        return liste
    first_element = liste[0]
    rest_list = liste[1:]
    if not isinstance(first_element, list):
        return [first_element] + flatten_nested_list(rest_list)
    else:
        return flatten_nested_list(first_element) + flatten_nested_list(rest_list)


if __name__ == "__main__":
    pass
