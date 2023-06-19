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
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from amorstructgp.nn.module import *
import numpy as np


class DatasetEncoder(nn.Module):
    """
    The following class is copied and then adapted from https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License

    Dataset Encoder of AHGP (Liu et al. 2020)
    """

    def __init__(self, num_attentions1, num_attentions2, att1_hidden_dim, att2_hidden_dim, dropout_p):
        super().__init__()
        self.input_dim = 2  # each datapoint x_i is splitted into single dimensions and the output is y_i is appended --> each element in the sequence has dimension=2
        self.num_attentions1 = num_attentions1
        self.num_attentions2 = num_attentions2
        self.att1_hidden_dim = att1_hidden_dim
        self.att2_hidden_dim = att2_hidden_dim
        self.input_projection = Linear(self.input_dim, self.att1_hidden_dim, bias=False)

        self.self_attentions1 = nn.ModuleList([Attention(self.att1_hidden_dim, dropout_p) for tt in range(self.num_attentions1)])

        self.self_attentions2 = nn.ModuleList([Attention(self.att2_hidden_dim, dropout_p) for tt in range(self.num_attentions2)])

    def forward(self, X_data_tr, F_data_tr, node_mask_tr, dim_mask, device=torch.device("cpu")):
        """
        X_data: B X N X D
        F_data: B X N
        initial node_mask: B X N
        X_input: (B X D) X N X 2
        dim_mask: B X D
        node_mask: B X N
        """
        # recale X
        # X_data_tr = torch.div(X_data_tr, 10.0)

        # preprocess data to each dimension
        batch_size = X_data_tr.shape[0]
        max_dim = X_data_tr.shape[2]
        max_num_data = X_data_tr.shape[1]
        f_tr = F_data_tr.unsqueeze(-1)  # B X N X 1
        f_expand = f_tr.expand(-1, -1, max_dim)  # B X N X D
        f_expand = f_expand * dim_mask.unsqueeze(-2)
        X_input = torch.cat((X_data_tr.unsqueeze(-1), f_expand.unsqueeze(-1)), -1)  # B X N X D X 2
        X_input = X_input.permute(0, 2, 1, 3)
        X_input = X_input.reshape(-1, max_num_data, 2)  # (B X D) X N X 2
        node_mask_tr = node_mask_tr.repeat(1, max_dim)  # B X (D X N)
        node_mask_tr = node_mask_tr.reshape(-1, max_num_data)  # (B X D) X N

        encoder_input = self.input_projection(X_input)
        # propagation
        for attention in self.self_attentions1:
            encoder_input, attns = attention(encoder_input, encoder_input, encoder_input, mask=node_mask_tr)
            # attns_list1.append(attns.detach())
            # encoder_input_list.append(encoder_input)

        N = torch.sum(node_mask_tr, dim=1)  # N: (B x D) X 0
        # node_encoder_mask = torch.repeat_interleave(node_mask.unsqueeze(-1),encoder_input.shape[-1],dim=-1)
        encoder_input = encoder_input * node_mask_tr.unsqueeze(-1)

        dim_encoder_input = torch.sum(encoder_input, 1) / (N.unsqueeze(-1))  # (B X D) X hidden_dim
        dim_encoder_input = dim_encoder_input.reshape(batch_size, max_dim, dim_encoder_input.shape[-1])  # B X D X hidden_dim

        # feed encoder_input to next attention network for dimension
        for attention in self.self_attentions2:
            dim_encoder_input, attns = attention(dim_encoder_input, dim_encoder_input, dim_encoder_input, mask=dim_mask)

        dim_encoder_mask = dim_mask.unsqueeze(-1)  # B X D X 1
        dimwise_dataset_encoding = dim_encoder_input * dim_encoder_mask  # B X D X hidden_dim

        return dimwise_dataset_encoding


class FullDimDatasetEncoder(nn.Module):
    def __init__(self, num_attentions1, num_attentions2, att1_hidden_dim, att2_hidden_dim, max_dim, dropout_p):
        super().__init__()
        # self.input_dim = 2  # each datapoint x_i is splitted into single dimensions and the output is y_i is appended --> each element in the sequence has dimension=2
        self.num_attentions1 = num_attentions1
        self.num_attentions2 = num_attentions2
        self.att1_hidden_dim = att1_hidden_dim
        self.att2_hidden_dim = att2_hidden_dim
        self.input_dim = max_dim
        self.input_projection = Linear(self.input_dim + 2, self.att1_hidden_dim, bias=False)

        self.self_attentions1 = nn.ModuleList([Attention(self.att1_hidden_dim, dropout_p) for tt in range(self.num_attentions1)])

        self.self_attentions2 = nn.ModuleList([Attention(self.att2_hidden_dim, dropout_p) for tt in range(self.num_attentions2)])

    def forward(self, X_data_tr, F_data_tr, node_mask_tr, dim_mask, device=torch.device("cpu")):
        """
        X_data: B X N X D
        F_data: B X N
        initial node_mask: B X N
        X_input: (B X D) X N X 2
        dim_mask: B X D
        node_mask: B X N
        """
        # recale X
        # X_data_tr = torch.div(X_data_tr, 10.0)

        # preprocess data to each dimension
        batch_size = X_data_tr.shape[0]
        max_dim = X_data_tr.shape[2]
        max_num_data = X_data_tr.shape[1]
        X_data_expanded = X_data_tr.unsqueeze(-2).expand(-1, -1, max_dim, -1)  # B x N x D x D
        X_data_expanded = X_data_expanded * dim_mask.unsqueeze(-2).unsqueeze(-1)
        padding_size = self.input_dim - max_dim
        X_data_expanded_padding = torch.zeros(batch_size, max_num_data, max_dim, padding_size, device=device)  # B x N x D x (INPUT_DIM - D)

        f_tr = F_data_tr.unsqueeze(-1)  # B X N X 1
        f_expand = f_tr.expand(-1, -1, max_dim)  # B X N X D
        f_expand = f_expand * dim_mask.unsqueeze(-2)
        X_input = torch.cat(
            (X_data_tr.unsqueeze(-1), f_expand.unsqueeze(-1), X_data_expanded, X_data_expanded_padding), -1
        )  # B X N X D X (2+INPUT_DIM)
        X_input = X_input.permute(0, 2, 1, 3)
        X_input = X_input.reshape(-1, max_num_data, 2 + self.input_dim)  # (B X D) X N X (2+INPUT_DIM)
        node_mask_tr = node_mask_tr.repeat(1, max_dim)  # B X (D X N)
        node_mask_tr = node_mask_tr.reshape(-1, max_num_data)  # (B X D) X N

        encoder_input = self.input_projection(X_input)
        # propagation
        for attention in self.self_attentions1:
            encoder_input, attns = attention(encoder_input, encoder_input, encoder_input, mask=node_mask_tr)
            # attns_list1.append(attns.detach())
            # encoder_input_list.append(encoder_input)

        N = torch.sum(node_mask_tr, dim=1)  # N: (B x D) X 0
        # node_encoder_mask = torch.repeat_interleave(node_mask.unsqueeze(-1),encoder_input.shape[-1],dim=-1)
        encoder_input = encoder_input * node_mask_tr.unsqueeze(-1)

        dim_encoder_input = torch.sum(encoder_input, 1) / (N.unsqueeze(-1))  # (B X D) X hidden_dim
        dim_encoder_input = dim_encoder_input.reshape(batch_size, max_dim, dim_encoder_input.shape[-1])  # B X D X hidden_dim

        # feed encoder_input to next attention network for dimension
        for attention in self.self_attentions2:
            dim_encoder_input, attns = attention(dim_encoder_input, dim_encoder_input, dim_encoder_input, mask=dim_mask)

        dim_encoder_mask = dim_mask.unsqueeze(-1)  # B X D X 1
        dimwise_dataset_encoding = dim_encoder_input * dim_encoder_mask  # B X D X hidden_dim

        return dimwise_dataset_encoding


class EnrichedDatasetEncoder(nn.Module):
    def __init__(
        self,
        num_attentions1,
        num_attentions2,
        num_attentions3,
        num_attentions4,
        att_hidden_dim,
        use_standard_layers,
        dim_intermediate_bert,
        dropout_p,
    ):
        super().__init__()
        self.input_dim = 2  # each datapoint x_i is splitted into single dimensions and the output is y_i is appended --> each element in the sequence has dimension=2
        self.num_attentions1 = num_attentions1
        self.num_attentions2 = num_attentions2
        self.num_attentions3 = num_attentions3
        self.num_attentions4 = num_attentions4
        self.att1_hidden_dim = att_hidden_dim
        self.att2_hidden_dim = att_hidden_dim
        self.att3_hidden_dim = att_hidden_dim * 2
        self.att4_hidden_dim = att_hidden_dim * 2

        self.input_projection = Linear(self.input_dim, self.att1_hidden_dim, bias=False)

        if use_standard_layers:
            self.self_attentions1 = nn.ModuleList(
                [AttentionLayer(self.att1_hidden_dim, dim_intermediate_bert, dropout_p) for tt in range(self.num_attentions1)]
            )

            self.self_attentions2 = nn.ModuleList(
                [AttentionLayer(self.att2_hidden_dim, dim_intermediate_bert, dropout_p) for tt in range(self.num_attentions2)]
            )

            self.self_attentions3 = nn.ModuleList(
                [AttentionLayer(self.att3_hidden_dim, dim_intermediate_bert, dropout_p) for tt in range(self.num_attentions3)]
            )

            self.self_attentions4 = nn.ModuleList(
                [AttentionLayer(self.att4_hidden_dim, dim_intermediate_bert, dropout_p) for tt in range(self.num_attentions4)]
            )
        else:
            self.self_attentions1 = nn.ModuleList([Attention(self.att1_hidden_dim, dropout_p) for tt in range(self.num_attentions1)])

            self.self_attentions2 = nn.ModuleList([Attention(self.att2_hidden_dim, dropout_p) for tt in range(self.num_attentions2)])

            self.self_attentions3 = nn.ModuleList([Attention(self.att3_hidden_dim, dropout_p) for tt in range(self.num_attentions3)])

            self.self_attentions4 = nn.ModuleList([Attention(self.att4_hidden_dim, dropout_p) for tt in range(self.num_attentions4)])

    def forward(self, X_data_tr, F_data_tr, node_mask, dim_mask, device=torch.device("cpu")):
        """
        X_data: B X N X D
        F_data: B X N
        initial node_mask: B X N
        X_input: (B X D) X N X 2
        dim_mask: B X D
        node_mask: B X N
        """
        # recale X
        # X_data_tr = torch.div(X_data_tr, 10.0)

        # preprocess data to each dimension
        batch_size = X_data_tr.shape[0]
        max_dim = X_data_tr.shape[2]
        max_num_data = X_data_tr.shape[1]
        f_tr = F_data_tr.unsqueeze(-1)  # B X N X 1
        f_expand = f_tr.expand(-1, -1, max_dim)  # B X N X D
        f_expand = f_expand * dim_mask.unsqueeze(-2)
        X_input = torch.cat((X_data_tr.unsqueeze(-1), f_expand.unsqueeze(-1)), -1)  # B X N X D X 2
        X_input = X_input.permute(0, 2, 1, 3)
        X_input = X_input.reshape(-1, max_num_data, 2)  # (B X D) X N X 2
        node_mask_tr = node_mask.repeat(1, max_dim)  # B X (D X N)
        node_mask_tr = node_mask_tr.reshape(-1, max_num_data)  # (B X D) X N

        # project 2dim input sequence element to bigger vector
        encoder_input = self.input_projection(X_input)  # (B x D) x N x hidden_dim

        #### In Dimension Transformer 1 ###############
        for attention in self.self_attentions1:
            encoder_input, attns = attention(encoder_input, encoder_input, encoder_input, mask=node_mask_tr)

        encoder_input = encoder_input * node_mask_tr.unsqueeze(-1)  # (B x D) x N x hidden_dim

        ##### Cross datapoint Transformer ####################
        encoder_input_expanded = encoder_input.reshape(batch_size, max_dim, max_num_data, encoder_input.shape[-1])  # B x D x N x hidden_dim

        encoder_input_expanded = encoder_input_expanded.permute(0, 2, 1, 3)  # B x N x D x hidden_dim

        dim_mask_tr = dim_mask.repeat(1, max_num_data)  # B x (N x D)
        dim_mask_tr = dim_mask_tr.reshape(batch_size, max_num_data, max_dim)  # B x N x D
        encoder_input_expanded = encoder_input_expanded * dim_mask_tr.unsqueeze(-1)  # B x N x D x hidden_dim

        D = torch.sum(dim_mask_tr, dim=2)  # B x N

        datapoint_encoder_input = torch.sum(encoder_input_expanded, 2) / D.unsqueeze(-1)  # B x N x hidden_dim

        for attention in self.self_attentions2:
            datapoint_encoder_input, attns = attention(
                datapoint_encoder_input, datapoint_encoder_input, datapoint_encoder_input, mask=node_mask
            )

        datapoint_encoding = datapoint_encoder_input * node_mask.unsqueeze(-1)  # B x N x hidden_dim

        datapoint_encoding = datapoint_encoding.repeat(1, 1, max_dim)  # B x N x (D x hidden_dim)

        datapoint_encoding = datapoint_encoding.reshape(
            batch_size, max_num_data, max_dim, encoder_input.shape[-1]
        )  # B x N x D x hidden_dim

        datapoint_encoding = datapoint_encoding.permute(0, 2, 1, 3)  # B x D x N x hidden_dim

        datapoint_encoding = datapoint_encoding.reshape(-1, max_num_data, encoder_input.shape[-1])  # (B x D) x N x hidden_dim

        encoder_input2 = torch.cat((encoder_input, datapoint_encoding), 2)  # (B x D) x N x 2*hidden_dim

        #### In Dimension Transformer 2 ###############
        for attention in self.self_attentions3:
            encoder_input2, attns = attention(encoder_input2, encoder_input2, encoder_input2, mask=node_mask_tr)

        encoder_input2 = encoder_input2 * node_mask_tr.unsqueeze(-1)  # (B x D) x N x 2*hidden_dim

        ##### Cross dimension Transformer #################
        N = torch.sum(node_mask_tr, dim=1)  # N: (B x D) X 0

        dim_encoder_input = torch.sum(encoder_input2, 1) / (N.unsqueeze(-1))  # (B X D) X 2*hidden_dim
        dim_encoder_input = dim_encoder_input.reshape(batch_size, max_dim, dim_encoder_input.shape[-1])  # B X D X 2*hidden_dim

        # feed encoder_input to next attention network for dimension
        for attention in self.self_attentions4:
            dim_encoder_input, attns = attention(dim_encoder_input, dim_encoder_input, dim_encoder_input, mask=dim_mask)

        dim_encoder_mask = dim_mask.unsqueeze(-1)  # B X D X 1
        dimwise_dataset_encoding = dim_encoder_input * dim_encoder_mask  # B X D X 2*hidden_dim

        return dimwise_dataset_encoding


def get_padded_dataset_and_masks(X_list: List[np.array], y_list: List[np.array]) -> Tuple[np.array, np.array, np.array]:
    sizes = [X.shape[0] for X in X_list]
    dims = [X.shape[1] for X in X_list]
    max_size = max(sizes)
    max_dim = max(dims)
    size_paddings = [max_size - size for size in sizes]
    dim_paddings = [max_dim - dim for dim in dims]
    X_padded = np.stack([np.pad(X, ((0, size_paddings[i]), (0, dim_paddings[i]))) for i, X in enumerate(X_list)])
    y_padded = np.stack([np.pad(y, (0, size_paddings[i])) for i, y in enumerate(y_list)])
    size_mask = np.stack([np.pad(np.ones(size), (0, size_paddings[i])) for i, size in enumerate(sizes)])
    dim_mask = np.stack([np.pad(np.ones(dim), (0, dim_paddings[i])) for i, dim in enumerate(dims)])
    diagonal_mask = np.stack(
        [np.pad(np.ones(size_paddings[i]), (size, 0), "constant", constant_values=0.0) for i, size in enumerate(sizes)]
    )
    size_mask_kernel = np.stack(
        [
            np.pad(np.ones((size, size)), ((0, size_paddings[i]), (0, size_paddings[i])), "constant", constant_values=0.0)
            for i, size in enumerate(sizes)
        ]
    )
    assert np.allclose(size_mask, 1.0 - diagonal_mask)
    assert np.allclose(1.0 - size_mask, diagonal_mask)
    N = np.array(sizes, dtype=np.float32)
    return X_padded, y_padded, size_mask, dim_mask, N, size_mask_kernel


if __name__ == "__main__":
    pass
