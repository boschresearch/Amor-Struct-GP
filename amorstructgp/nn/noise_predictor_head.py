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
from typing import List
import torch
import torch.nn as nn
from amorstructgp.config.prior_parameters import NOISE_VARIANCE_EXPONENTIAL_LAMBDA
from amorstructgp.gp.base_kernels import BaseKernelEvalMode
from amorstructgp.nn.module import Linear
import torch.nn.functional as F


class BaseNoiseVariancePredictorHead(nn.Module):
    def __init__(self, noise_variance_lower_bound: float, use_scaled_softplus: bool) -> None:
        super().__init__()
        self.noise_variance_lower_bound = noise_variance_lower_bound
        self.noise_variance_prior = torch.distributions.Exponential(NOISE_VARIANCE_EXPONENTIAL_LAMBDA)
        self.eval_mode = BaseKernelEvalMode.STANDARD
        self.debug_variance = 1e-2
        self.use_scaled_softplus = use_scaled_softplus
        if self.use_scaled_softplus:
            self.scale_beta = 2
            self.scale_a = 0.1
        else:
            self.scale_beta = 1
            self.scale_a = 1.0
        self.softplus = nn.Softplus(beta=self.scale_beta)

    @abstractmethod
    def forward(
        self, kernel_embeddings: torch.Tensor, kernel_mask: torch.Tensor, dimwise_dataset_encoding: torch.Tensor, dim_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        kernel_embeddings B X D X N_k x d_h (output of kernel encoder)
        kernel_mask B X D x N_k
        dimwise_dataset_encoding B X D X hidden_dim (output of dataset encoder)
        dim_mask: B X D

        Output:
        torch.Tensor B x 1 - noise variances
        """
        raise NotImplementedError

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode

    def greater_than(self, x, lower_bound: float):
        return self.scale_a * self.softplus(x) + lower_bound

    def get_n_base_kernels(self, kernel_mask):
        # Extracts number of base kernels in each batch - this is a sum over all active dimension and over the list of base kernels in each dimension (base_kernel=symbol)
        n_kernels = torch.sum(kernel_mask, dim=[1, 2])
        n_kernels = n_kernels.unsqueeze(-1)
        return n_kernels

    def get_n_dims(self, dim_mask):
        # Extract number of active dimension for each batch element
        n_dims = torch.sum(dim_mask, dim=1)  # B
        n_dims = n_dims.unsqueeze(-1)  # B x 1
        return n_dims


class NoiseVariancePredictorHead(BaseNoiseVariancePredictorHead):
    def __init__(
        self,
        dim_hidden_layer_list: List[int],
        kernel_embedding_dim: int,
        noise_variance_lower_bound: float,
        dropout_p: float,
        use_scaled_softplus: bool,
        **kwargs
    ) -> None:
        super().__init__(noise_variance_lower_bound, use_scaled_softplus)

        assert len(dim_hidden_layer_list) > 0
        self.dropout = nn.Dropout(p=dropout_p)
        use_bias = True
        self.mlp_linear_layers = []
        for i, dim_hidden in enumerate(dim_hidden_layer_list):
            if i == 0:
                linear_layer = Linear(kernel_embedding_dim, dim_hidden, bias=use_bias)
            else:
                linear_layer = Linear(dim_hidden_layer_list[i - 1], dim_hidden, bias=use_bias)
            self.mlp_linear_layers.append(linear_layer)

        assert len(self.mlp_linear_layers) == len(dim_hidden_layer_list)
        self.mlp_linear_layers = nn.ModuleList(self.mlp_linear_layers)
        self.final_layer = Linear(dim_hidden_layer_list[-1], 1, bias=use_bias)

    def forward(self, kernel_embeddings, kernel_mask, dimwise_dataset_encoding, dim_mask):
        # kernel_embeddings B X D X N_k x d_h (output of kernel encoder)
        # kernel_mask B X D x N_k
        n_kernels = self.get_n_base_kernels(kernel_mask)  # B x 1
        # average over all kernel embeddings in a batch to create a global embedding of the kernel
        global_kernel_embedding = torch.sum(kernel_embeddings, dim=[1, 2]) / n_kernels  # B x d_h
        hidden = global_kernel_embedding
        for linear_layer in self.mlp_linear_layers:
            hidden = F.relu(linear_layer(hidden))
            hidden = self.dropout(hidden)
        untransformed_noise = self.final_layer(hidden)  # B x 1
        noise_variance = self.greater_than(untransformed_noise, lower_bound=self.noise_variance_lower_bound)  # B x 1
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            batch_size = kernel_embeddings.shape[0]
            noise_variance = torch.tensor(self.debug_variance)
            noise_variance = noise_variance.repeat(batch_size, 1)  # B x 1
            untransformed_noise = None
        log_prior_prob = torch.squeeze(self.noise_variance_prior.log_prob(noise_variance), 1)  # B
        return noise_variance, untransformed_noise, log_prior_prob


class NoiseVariancePredictorHeadWithDatasetEncoding(BaseNoiseVariancePredictorHead):
    def __init__(
        self,
        dim_hidden_layer_list: List[int],
        kernel_embedding_dim: int,
        dataset_encoding_dim: int,
        noise_variance_lower_bound: float,
        dropout_p: float,
        use_scaled_softplus: bool,
        **kwargs
    ) -> None:
        super().__init__(noise_variance_lower_bound, use_scaled_softplus)

        assert len(dim_hidden_layer_list) > 0
        self.dropout = nn.Dropout(p=dropout_p)
        use_bias = True
        dim_input_layer = kernel_embedding_dim + dataset_encoding_dim
        self.mlp_linear_layers = []
        for i, dim_hidden in enumerate(dim_hidden_layer_list):
            if i == 0:
                linear_layer = Linear(dim_input_layer, dim_hidden, bias=use_bias)
            else:
                linear_layer = Linear(dim_hidden_layer_list[i - 1], dim_hidden, bias=use_bias)
            self.mlp_linear_layers.append(linear_layer)

        assert len(self.mlp_linear_layers) == len(dim_hidden_layer_list)
        self.mlp_linear_layers = nn.ModuleList(self.mlp_linear_layers)
        self.final_layer = Linear(dim_hidden_layer_list[-1], 1, bias=use_bias)

    def forward(self, kernel_embeddings, kernel_mask, dimwise_dataset_encoding, dim_mask):
        # kernel_embeddings B X D X N_k x d_h (output of kernel encoder)
        # kernel_mask B X D x N_k
        # dimwise_dataset_encoding B X D X hidden_dim (output of dataset encoder)
        # dim_mask: B X D
        n_kernels = self.get_n_base_kernels(kernel_mask)  # B x 1
        # average over all kernel embeddings in a batch to create a global embedding of the kernel
        global_kernel_embedding = torch.sum(kernel_embeddings, dim=[1, 2]) / n_kernels  # B x d_h

        n_dims = self.get_n_dims(dim_mask)
        # average over dimwise dataset embeddings to create global dataset embedding
        global_dataset_embedding = torch.sum(dimwise_dataset_encoding, dim=1) / n_dims  # B x hidden_dim
        hidden = torch.cat((global_kernel_embedding, global_dataset_embedding), dim=1)  # B x dim_input_layer
        for linear_layer in self.mlp_linear_layers:
            hidden = F.relu(linear_layer(hidden))
            hidden = self.dropout(hidden)
        untransformed_noise = self.final_layer(hidden)  # B x 1
        noise_variance = self.greater_than(untransformed_noise, lower_bound=self.noise_variance_lower_bound)  # B x 1
        if self.eval_mode == BaseKernelEvalMode.DEBUG:
            batch_size = kernel_embeddings.shape[0]
            noise_variance = torch.tensor(self.debug_variance)
            noise_variance = noise_variance.repeat(batch_size, 1)  # B x 1
            untransformed_noise = None
        log_prior_prob = torch.squeeze(self.noise_variance_prior.log_prob(noise_variance), 1)  # B
        return noise_variance, untransformed_noise, log_prior_prob
