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
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from amorstructgp.nn.dataset_encoder import DatasetEncoder
from amorstructgp.nn.module import *
import numpy as np
from enum import Enum


class BaseKernelEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dataset_enc_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim  # d_h
        self.dataset_enc_dim = dataset_enc_dim  # d_d
        self.input_dim = input_dim  # d_k

    @abstractmethod
    def forward(
        self, kernel_embeddings: torch.Tensor, dataset_embeddings: torch.Tensor, kernel_mask: torch.Tensor, dim_mask: torch.Tensor
    ) -> torch.Tensor:
        # kernel_embeddings B X D X N_k X d_k
        # dataset_embeddings B X D x d_d
        # kernel_mask B X D x N_k
        # dim_mask B x D
        # Output: kernel_embeddings B X D X N_k x d_h
        raise NotImplementedError


class KernelEncoder(BaseKernelEncoder):
    def __init__(self, input_dim: int, n_enc_dec_layer: int, hidden_dim: int, dataset_enc_dim: int, dropout_p: float, **kwargs):
        super().__init__(input_dim, hidden_dim, dataset_enc_dim)
        self.n_enc_dec_layer = n_enc_dec_layer
        self.input_projection = Linear(self.input_dim, self.hidden_dim, bias=False)

        # Initialize layers
        self.layers = nn.ModuleList(
            [KernelEncodingLayer(self.hidden_dim, self.dataset_enc_dim, dropout_p) for i in range(self.n_enc_dec_layer)]
        )

    def forward(self, kernel_embeddings, dataset_embeddings, kernel_mask, dim_mask):
        # kernel_embeddings B X D X N_k X d_k
        # dataset_embeddings B X D x d_d
        # kernel_mask B X D x N_k
        # dim_mask B x D
        batch_size = kernel_embeddings.size(0)
        max_dim = kernel_embeddings.size(1)
        max_len_kernels = kernel_embeddings.size(2)
        original_kernel_mask = kernel_mask
        # Collate Batch and dimensions
        kernel_embeddings = kernel_embeddings.reshape(-1, max_len_kernels, self.input_dim)  # (B X D) X N_k x d_k
        dataset_embeddings = dataset_embeddings.reshape(-1, self.dataset_enc_dim)  # (B x D) x d_d

        kernel_mask = kernel_mask.reshape(-1, max_len_kernels)  # (B x D) x N_k

        kernel_embeddings = self.input_projection(kernel_embeddings)  # (B X D) X N_k x d_h

        for layer in self.layers:
            kernel_embeddings, attns = layer(kernel_embeddings, kernel_embeddings, kernel_embeddings, dataset_embeddings, mask=kernel_mask)

        kernel_embeddings = kernel_embeddings.reshape(batch_size, max_dim, max_len_kernels, self.hidden_dim)  # B X D X N_k x d_h

        kernel_embeddings = kernel_embeddings * original_kernel_mask.unsqueeze(-1)

        return kernel_embeddings


class CrossAttentionKernelEncoder(BaseKernelEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dataset_enc_dim: int,
        n_enc_layer,
        n_cross_layer,
        n_dec_layer,
        use_standard_layers,
        dropout_p,
        **kwargs
    ):
        super().__init__(input_dim, hidden_dim, dataset_enc_dim)
        self.n_enc_layer = n_enc_layer
        self.n_cross_layer = n_cross_layer
        self.n_dec_layer = n_dec_layer

        self.input_projection = Linear(self.input_dim, self.hidden_dim, bias=False)
        if use_standard_layers:
            num_intermediate = 2 * self.hidden_dim
            self.layers_enc = nn.ModuleList(
                [
                    KernelEncodingStandardLayer(self.hidden_dim, num_intermediate, self.dataset_enc_dim, dropout_p)
                    for i in range(self.n_enc_layer)
                ]
            )
            self.layers_cross = nn.ModuleList(
                [AttentionLayer(self.hidden_dim, num_intermediate, dropout_p) for i in range(self.n_cross_layer)]
            )
            self.layers_dec = nn.ModuleList(
                [
                    KernelEncodingStandardLayer(self.hidden_dim, num_intermediate, self.dataset_enc_dim + self.hidden_dim, dropout_p)
                    for i in range(self.n_dec_layer)
                ]
            )
        else:
            self.layers_enc = nn.ModuleList(
                [KernelEncodingLayer(self.hidden_dim, self.dataset_enc_dim, dropout_p) for i in range(self.n_enc_layer)]
            )
            self.layers_cross = nn.ModuleList([Attention(self.hidden_dim, dropout_p) for i in range(self.n_cross_layer)])
            self.layers_dec = nn.ModuleList(
                [KernelEncodingLayer(self.hidden_dim, self.dataset_enc_dim + self.hidden_dim, dropout_p) for i in range(self.n_dec_layer)]
            )

    def forward(
        self, kernel_embeddings: torch.Tensor, dataset_embeddings: torch.Tensor, kernel_mask: torch.Tensor, dim_mask: torch.Tensor
    ) -> torch.Tensor:
        # kernel_embeddings B X D X N_k X d_k
        # dataset_embeddings B X D x d_d
        # kernel_mask B X D x N_k
        # dim_mask B x D
        batch_size = kernel_embeddings.size(0)
        max_dim = kernel_embeddings.size(1)
        max_len_kernels = kernel_embeddings.size(2)
        original_kernel_mask = kernel_mask
        # Collate Batch and dimensions
        kernel_embeddings = kernel_embeddings.reshape(-1, max_len_kernels, self.input_dim)  # (B X D) X N_k x d_k
        original_dataset_embeddings = dataset_embeddings  # B X D x d_d
        dataset_embeddings = dataset_embeddings.reshape(-1, self.dataset_enc_dim)  # (B x D) x d_d

        kernel_mask = kernel_mask.reshape(-1, max_len_kernels)  # (B x D) x N_k

        kernel_embeddings = self.input_projection(kernel_embeddings)  # (B X D) X N_k x d_h

        #### kernel encoding inside each dimension ##########
        for layer in self.layers_enc:
            kernel_embeddings, attns = layer(kernel_embeddings, kernel_embeddings, kernel_embeddings, dataset_embeddings, mask=kernel_mask)

        kernel_embeddings = kernel_embeddings.reshape(batch_size, max_dim, max_len_kernels, self.hidden_dim)  # B X D X N_k x d_h

        kernel_embeddings = kernel_embeddings * original_kernel_mask.unsqueeze(-1)  # B X D X N_k x d_h

        N_k = torch.sum(original_kernel_mask, 2)  # B x D
        # make sure you dont divide by zero
        N_k = torch.clamp(N_k, min=0.1)

        cross_embeddings = torch.sum(kernel_embeddings, 2) / N_k.unsqueeze(-1)  # B x D x d_h

        #### cross attention between kernel encoding over dimension ##########
        for layer in self.layers_cross:
            cross_embeddings, attns = layer(cross_embeddings, cross_embeddings, cross_embeddings, mask=dim_mask)

        dim_encoder_mask = dim_mask.unsqueeze(-1)  # B X D X 1
        cross_embeddings = cross_embeddings * dim_encoder_mask  # B x D x d_h

        combined_dataset_cross_embedding = torch.cat((original_dataset_embeddings, cross_embeddings), dim=2)  # B x D x (d_d + d_h)

        combined_dataset_cross_embedding = combined_dataset_cross_embedding.reshape(
            -1, self.dataset_enc_dim + self.hidden_dim
        )  # (B x D) x (d_d + d_h)

        kernel_embeddings = kernel_embeddings.reshape(-1, max_len_kernels, self.hidden_dim)  # (B x D) x N_k x d_h

        #### kernel decoding to final embedding/parameters inside each dimension ##########
        for layer in self.layers_dec:
            kernel_embeddings, attns = layer(
                kernel_embeddings, kernel_embeddings, kernel_embeddings, combined_dataset_cross_embedding, mask=kernel_mask
            )

        kernel_embeddings = kernel_embeddings.reshape(batch_size, max_dim, max_len_kernels, self.hidden_dim)  # B X D X N_k x d_h

        kernel_embeddings = kernel_embeddings * original_kernel_mask.unsqueeze(-1)  # B X D X N_k x d_h

        return kernel_embeddings


if __name__ == "__main__":
    pass
