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
from typing import List, Optional
import torch
import torch.nn as nn
from amorstructgp.config.prior_parameters import KERNEL_LENGTHSCALE_GAMMA, KERNEL_VARIANCE_GAMMA
from amorstructgp.gp.base_kernels import (
    BaseKernel,
    BaseKernelEvalMode,
    BaseKernelParameterFormat,
    BatchedDimWiseAdditiveKernelWrapper,
    KernelParameterNestedList,
    KernelTypeList,
    squared_dist,
)
from amorstructgp.nn.module import MLP


class SEKernelOnlyLengthscale(nn.Module, BaseKernel):
    def __init__(self, kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode=BaseKernelEvalMode.STANDARD) -> None:
        super().__init__()
        self.num_params = 1
        self.register_buffer("a_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[0]))
        self.register_buffer("b_lengthscale", torch.tensor(KERNEL_LENGTHSCALE_GAMMA[1]))
        self.mlp = MLP(kernel_embedding_dim, dim_hidden_layer_list, self.num_params, dropout_p=dropout_p, use_biases=True)
        self.softplus = torch.nn.Softplus()
        self.lengthscale_index = 0
        self.eval_mode = eval_mode
        self.debug_lengthscale = torch.tensor(1.0)

    def get_lengthscale_prior(self):
        return torch.distributions.Gamma(self.a_lengthscale, self.b_lengthscale)

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode

    def forward(self, X1, X2, kernel_embedding, untransformed_params=None):
        # x1: N1 x D
        lengthscale = self._get_parameters(kernel_embedding, untransformed_params)
        sq_distances = squared_dist(X1, X2, lengthscale, 0)
        K = torch.exp(-0.5 * sq_distances)
        return K

    def _get_parameters(self, kernel_embedding, untransformed_params):
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            lengthscale = self.debug_lengthscale
        elif self.eval_mode == BaseKernelEvalMode.STANDARD:
            params = self.softplus(self.mlp(kernel_embedding))
            lengthscale = params[self.lengthscale_index]
        elif self.eval_mode == BaseKernelEvalMode.WARM_START:
            params = self.softplus(untransformed_params)
            lengthscale = params[self.lengthscale_index]
        elif self.eval_mode == BaseKernelEvalMode.VALUE_LIST:
            params = untransformed_params
            lengthscale = params[self.lengthscale_index]
        return lengthscale

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        lengthscale = self._get_parameters(kernel_embedding, untransformed_params)
        return self._get_log_prior_value(lengthscale)

    def _get_log_prior_value(self, lengthscale):
        log_p_lengthscale = self.get_lengthscale_prior().log_prob(lengthscale)
        return log_p_lengthscale

    def get_untransformed_parameters(self, kernel_embedding):
        return self.mlp(kernel_embedding).clone().detach().requires_grad_(True)

    def get_parameters(self, kernel_embedding, detach: bool = True):
        if detach:
            return self.softplus(self.mlp(kernel_embedding)).clone().detach().requires_grad_(True)
        else:
            return self.softplus(self.mlp(kernel_embedding))

    def get_num_params(self) -> int:
        return self.num_params


class ConstantKernel(nn.Module, BaseKernel):
    def __init__(self, kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode=BaseKernelEvalMode.STANDARD) -> None:
        super().__init__()
        self.num_params = 1
        self.register_buffer("a_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[0]))
        self.register_buffer("b_variance", torch.tensor(KERNEL_VARIANCE_GAMMA[1]))
        self.mlp = MLP(kernel_embedding_dim, dim_hidden_layer_list, self.num_params, dropout_p=dropout_p, use_biases=True)
        self.softplus = torch.nn.Softplus()
        self.variance_index = 0
        self.eval_mode = eval_mode
        self.debug_variance = torch.tensor(1.0)

    def get_variance_prior(self):
        return torch.distributions.Gamma(self.a_variance, self.b_variance)

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode

    def forward(self, X1, X2, kernel_embedding, untransformed_params=None):
        # x1: N1 x D
        n1 = len(X1)
        n2 = len(X2)
        variance = self._get_parameters(kernel_embedding, untransformed_params)
        K = torch.pow(variance, 2.0)  # * torch.full((n1, n2), 1.0)
        return K

    def _get_parameters(self, kernel_embedding, untransformed_params):
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            variance = self.debug_variance
        elif self.eval_mode == BaseKernelEvalMode.STANDARD:
            params = self.softplus(self.mlp(kernel_embedding))
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.WARM_START:
            params = self.softplus(untransformed_params)
            variance = params[self.variance_index]
        elif self.eval_mode == BaseKernelEvalMode.VALUE_LIST:
            params = untransformed_params
            variance = params[self.variance_index]
        return variance

    def get_log_prior_prob(
        self, kernel_embedding: torch.tensor, untransformed_params: Optional[BaseKernelParameterFormat] = None
    ) -> torch.tensor:
        variance = self._get_parameters(kernel_embedding, untransformed_params)
        return self._get_log_prior_value(variance)

    def _get_log_prior_value(self, variance):
        log_p_variance = self.get_variance_prior().log_prob(variance)
        return log_p_variance

    def get_untransformed_parameters(self, kernel_embedding):
        return self.mlp(kernel_embedding).clone().detach().requires_grad_(True)

    def get_parameters(self, kernel_embedding, detach: bool = True):
        if detach:
            return self.softplus(self.mlp(kernel_embedding)).clone().detach().requires_grad_(True)
        else:
            return self.softplus(self.mlp(kernel_embedding))

    def get_num_params(self) -> int:
        return self.num_params


class RBFKernelWrapper(nn.Module):
    """
    Module that can calculate the gram matrix of the RBF kernel - has same interface as DimWiseAdditiveKernelWrapper.
    """

    def __init__(
        self,
        kernel_embedding_dim: int,
        dim_hidden_layer_list: List[int],
        dropout_p: float,
        eval_mode=BaseKernelEvalMode.STANDARD,
    ):
        super().__init__()
        self.eval_mode = eval_mode
        self.se_kernel = SEKernelOnlyLengthscale(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode)
        self.const_kernel = ConstantKernel(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode)
        self.eval_param_list = lambda params, d, i: None if params is None else params[d][i]

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode
        self.se_kernel.set_eval_mode(eval_mode)
        self.const_kernel.set_eval_mode(eval_mode)

    def forward(
        self,
        X1: torch.tensor,
        X2: torch.tensor,
        kernel_embeddings: torch.tensor,
        kernel_type_list: Optional[KernelTypeList],
        untransformed_params: Optional[KernelParameterNestedList] = None,
    ):
        """
        Arguments:
            X1: torch.tensor with shape N1 x D
            X2: torch.tensor with shape N2 x D
            kernel_embeddings: torch.tensor with shape D x 1 x d_k
            kernel_type_list: ignored
            untransformed_params: @TODO: add doc
        """

        kernel_grams = []
        D = len(kernel_type_list)
        for d in range(0, D):
            kernel_gram_at_d = self.se_kernel.forward(
                X1[:, d].unsqueeze(-1), X2[:, d].unsqueeze(-1), kernel_embeddings[d, 0, :], self.eval_param_list(untransformed_params, d, 0)
            )
            kernel_grams.append(kernel_gram_at_d)
        global_kernel_embedding = self.get_global_kernel_embedding(kernel_embeddings, D)
        constant = self.const_kernel.forward(X1, X2, global_kernel_embedding, self.eval_param_list(untransformed_params, d + 1, 0))
        # kernel_grams.append(constant_gram)
        K = constant * torch.prod(torch.stack(kernel_grams), dim=0)  # N1 x N2
        return K

    def get_global_kernel_embedding(self, kernel_embeddings, D):
        stacked_kernel_embeddings = torch.stack([kernel_embeddings[d, 0, :] for d in range(0, D)])
        avg_kernel_embedding = torch.squeeze(torch.mean(stacked_kernel_embeddings, dim=0))
        return avg_kernel_embedding

    def get_log_prior_prob(
        self,
        kernel_embeddings: torch.tensor,
        kernel_type_list: KernelTypeList = None,
        untransformed_params: Optional[KernelParameterNestedList] = None,
    ) -> torch.tensor:
        """
        Method to get the log prior prob of the kernel parameters that are given when evaluating the kernel

        Arguments:
            kernel_embeddings: torch.tensor with shape D x N_k x d_k
            kernel_type_list: ignored
            untransformed_params: pass
        """
        log_prior_probs = []
        D = len(kernel_type_list)
        for d in range(0, D):
            log_prior_probs_per_dim = self.se_kernel.get_log_prior_prob(
                kernel_embeddings[d, 0, :], self.eval_param_list(untransformed_params, d, 0)
            )
            log_prior_probs.append(log_prior_probs_per_dim)
        global_kernel_embedding = self.get_global_kernel_embedding(kernel_embeddings, D)
        log_prior_prob_variance = self.const_kernel.get_log_prior_prob(
            global_kernel_embedding, self.eval_param_list(untransformed_params, d + 1, 0)
        )
        log_prior_probs.append(log_prior_prob_variance)
        log_prior_prob = torch.sum(torch.stack(log_prior_probs), dim=0)
        return log_prior_prob

    def get_num_params(self, kernel_type_list: KernelTypeList) -> int:
        D = len(kernel_type_list)
        return D + 1

    def get_untransformed_parameters(self, kernel_embeddings: torch.tensor, kernel_type_list: KernelTypeList):
        untransformed_kernel_parms = []
        D = len(kernel_type_list)
        for d in range(0, D):
            untransformed_kernel_parms_per_dim = [self.se_kernel.get_untransformed_parameters(kernel_embeddings[d, 0, :])]
            untransformed_kernel_parms.append(untransformed_kernel_parms_per_dim)
        global_kernel_embedding = self.get_global_kernel_embedding(kernel_embeddings, D)
        untransformed_kernel_parms.append([self.const_kernel.get_untransformed_parameters(global_kernel_embedding)])
        return untransformed_kernel_parms

    def get_parameters(self, kernel_embeddings: torch.tensor, kernel_type_list: KernelTypeList, detach: bool = True):
        kernel_parms = []
        D = len(kernel_type_list)
        for d in range(0, D):
            kernel_parms_per_dim = [self.se_kernel.get_parameters(kernel_embeddings[d, 0, :], detach)]
            kernel_parms.append(kernel_parms_per_dim)
        global_kernel_embedding = self.get_global_kernel_embedding(kernel_embeddings, D)
        kernel_parms.append([self.const_kernel.get_parameters(global_kernel_embedding, detach)])
        return kernel_parms


class BatchedRBFKernelWrapper(BatchedDimWiseAdditiveKernelWrapper):
    """
    Batch version of RBFKernelWrapper
    """

    def __init__(
        self,
        kernel_embedding_dim: int,
        dim_hidden_layer_list: List[int],
        dropout_p: float,
        eval_mode=BaseKernelEvalMode.STANDARD,
    ):
        super().__init__(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, True, eval_mode)
        self.eval_mode = eval_mode
        self.dim_wise_additive_kernel_wrapper = RBFKernelWrapper(kernel_embedding_dim, dim_hidden_layer_list, dropout_p, eval_mode)
        self.eval_param_list = lambda params, i: None if params is None else params[i]
