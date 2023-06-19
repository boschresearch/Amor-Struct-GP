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
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Linear(nn.Module):
    """
    This class is copied and then adapted from https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init="linear"):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class MLP(nn.Module):
    """
    Standard MLP
    """

    def __init__(self, in_dim: int, dim_hidden_layer_list: List[int], out_dim: int, dropout_p=0.1, use_biases=True):
        super().__init__()
        self.dropout_p = dropout_p
        self.mlp_linear_layers = []
        self.dropout = nn.Dropout(p=dropout_p)
        for i, dim_hidden in enumerate(dim_hidden_layer_list):
            if i == 0:
                linear_layer = Linear(in_dim, dim_hidden, bias=use_biases)
            else:
                linear_layer = Linear(dim_hidden_layer_list[i - 1], dim_hidden, bias=use_biases)
            self.mlp_linear_layers.append(linear_layer)

        assert len(self.mlp_linear_layers) == len(dim_hidden_layer_list)
        self.mlp_linear_layers = nn.ModuleList(self.mlp_linear_layers)
        if len(dim_hidden_layer_list) == 0:
            self.final_layer = Linear(in_dim, out_dim, bias=use_biases)
        else:
            self.final_layer = Linear(dim_hidden_layer_list[-1], out_dim, bias=use_biases)

    def forward(self, input):
        hidden = input
        for linear_layer in self.mlp_linear_layers:
            hidden = F.relu(linear_layer(hidden))
            hidden = self.dropout(hidden)
        output = self.final_layer(hidden)
        return output


class MultiheadAttention(nn.Module):
    """
    This class is copied and then adapted from https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k, dropout_p=0.1):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=dropout_p)

    def forward(self, key, value, query, mask=None):
        # Get attention score
        # query, key, value: B x h x N x dv
        attn = torch.matmul(query, key.transpose(2, 3))  # B x h x N x N
        attn = attn / math.sqrt(self.num_hidden_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)
        # Dropout
        attn = self.attn_dropout(attn)
        # Get Context Vector
        result = torch.matmul(attn, value)

        return result, attn


class AttentionLayer(nn.Module):
    """
    This class is copied and then adapted from https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License
    Attention layer Standard
    """

    def __init__(self, num_hidden, num_intermediate, dropout_p=0.1, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(AttentionLayer, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn, dropout_p)

        self.attn_output_dropout = nn.Dropout(p=dropout_p)
        self.intermediate_dropout = nn.Dropout(p=dropout_p)
        self.final_output_dropout = nn.Dropout(p=dropout_p)

        self.attn_linear = Linear(num_hidden, num_hidden)
        self.intermedia_linear = Linear(num_hidden, num_intermediate)
        self.final_linear = Linear(num_intermediate, num_hidden)

        self.attention_layer_norm = nn.LayerNorm(num_hidden)
        self.output_layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query, mask=None):
        residual = value
        # Multi-head-attntion-block
        attn_output, attns = self.sa_block(key, value, query, mask)
        # residual connection and layernorm
        attn_output = self.attention_layer_norm(attn_output + residual)  # B X N X d
        # MLP
        final_output = self.mlp_block(attn_output)  # B X N X d
        # residual connection and layernorm
        final_output = self.output_layer_norm(final_output + attn_output)
        return final_output, attns

    def mlp_block(self, attn_output):
        # intermediate linear layer and activation
        intermediate_output = F.gelu(self.intermedia_linear(attn_output))
        # drop out intermediate layer
        intermediate_output = self.intermediate_dropout(intermediate_output)
        # Final linear
        final_output = self.final_linear(intermediate_output)
        # Residual dropout & connection
        final_output = self.final_output_dropout(final_output)
        return final_output

    def sa_block(self, key, value, query, mask):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        seq_v = value.size(1)
        # Make multihead: B x N x h x dv
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_v, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        # Transpose for attention dot product: B x h x N x dv
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # B x N --> B x 1 x 1 x N

        # Get context vector
        attn_output, attns = self.multihead(key, value, query, mask=mask)
        # Concatenate all multihead context vector
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_k, -1)  # B X N X d
        # linear layer after attention
        attn_output = self.attn_linear(attn_output)
        attn_output = self.attn_output_dropout(attn_output)
        return attn_output, attns


class Attention(nn.Module):
    """
    This class is copied and then adapted from https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License
    Attention Layer used in Tranformer
    """

    def __init__(self, num_hidden, dropout_p=0.1, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn, dropout_p)

        self.residual_dropout = nn.Dropout(p=dropout_p)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        seq_v = value.size(1)
        residual = value

        # Make multihead: B x N x h x dv
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_v, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        # Transpose for attention dot product: B x h x N x dv
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # B x N --> B x 1 x 1 x N

        # Get context vector
        result, attns = self.multihead(key, value, query, mask=mask)
        # Concatenate all multihead context vector
        result = result.transpose(1, 2).contiguous().view(batch_size, seq_k, -1)  # B X N X d

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)

        # Final linear
        result = F.relu(self.final_linear(result))

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual
        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class KernelEncodingLayer(nn.Module):
    def __init__(self, num_hidden, dataset_encoding_dim, dropout_p=0.1, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(KernelEncodingLayer, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn, dropout_p)

        self.residual_dropout = nn.Dropout(p=dropout_p)

        self.final_linear = Linear(num_hidden * 2 + dataset_encoding_dim, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query, dataset_enc, mask=None):
        # key,value,query: B x N x d
        # dataset_enc: B x d_enc
        # mask: B x N

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        seq_v = value.size(1)
        residual = value

        dataset_enc = dataset_enc.unsqueeze(1)  # B x 1 x d_enc

        dataset_enc = dataset_enc.repeat(1, seq_v, 1)  # B x N x d_enc

        if mask is not None:
            dataset_enc = dataset_enc * mask.unsqueeze(-1)

        # Make multihead: B x N x h x dv
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_v, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        # Transpose for attention dot product: B x h x N x dv
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # B x N --> B x 1 x 1 x N

        # Get context vector
        result, attns = self.multihead(key, value, query, mask=mask)
        # Concatenate all multihead context vector
        result = result.transpose(1, 2).contiguous().view(batch_size, seq_k, -1)  # B X N X d

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result, dataset_enc], dim=-1)

        # Final linear
        result = F.relu(self.final_linear(result))

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual
        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class KernelEncodingStandardLayer(nn.Module):
    """
    Standard Kernel Encoding layer
    """

    def __init__(self, num_hidden, num_intermediate, context_encoding_dim, dropout_p=0.1, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(KernelEncodingStandardLayer, self).__init__()

        self.num_hidden = num_hidden
        self.context_encoding_dim = context_encoding_dim
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn, dropout_p)

        self.attn_output_dropout = nn.Dropout(p=dropout_p)
        self.intermediate_dropout = nn.Dropout(p=dropout_p)
        self.final_output_dropout = nn.Dropout(p=dropout_p)

        self.attn_linear = Linear(num_hidden, num_hidden)
        self.intermedia_linear = Linear(num_hidden + self.context_encoding_dim, num_intermediate)
        self.final_linear = Linear(num_intermediate, num_hidden)

        self.attention_layer_norm = nn.LayerNorm(num_hidden)
        self.output_layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query, context_encoding, mask=None):
        # key,value,query: B x N x d
        # context_encoding: B x d_enc
        # mask: B x N
        seq_v = value.size(1)
        context_encoding = context_encoding.unsqueeze(1)  # B x 1 x d_enc
        context_encoding = context_encoding.repeat(1, seq_v, 1)  # B x N x d_enc
        if mask is not None:
            context_encoding = context_encoding * mask.unsqueeze(-1)

        residual = value
        # Multi-head-attntion-block
        attn_output, attns = self.sa_block(key, value, query, mask)
        # residual connection and layernorm
        attn_output = self.attention_layer_norm(attn_output + residual)  # B X N X d
        # MLP
        final_output = self.mlp_block(attn_output, context_encoding)  # B X N X d
        # residual connection and layernorm
        final_output = self.output_layer_norm(final_output + attn_output)
        return final_output, attns

    def mlp_block(self, attn_output, context_encoding):
        # concatenate context
        concatenated = torch.cat([attn_output, context_encoding], dim=-1)
        # intermediate linear layer and activation
        intermediate_output = F.gelu(self.intermedia_linear(concatenated))
        # drop out intermediate layer
        intermediate_output = self.intermediate_dropout(intermediate_output)
        # Final linear
        final_output = self.final_linear(intermediate_output)
        # Residual dropout & connection
        final_output = self.final_output_dropout(final_output)
        return final_output

    def sa_block(self, key, value, query, mask):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        seq_v = value.size(1)
        # Make multihead: B x N x h x dv
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_v, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        # Transpose for attention dot product: B x h x N x dv
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # B x N --> B x 1 x 1 x N

        # Get context vector
        attn_output, attns = self.multihead(key, value, query, mask=mask)
        # Concatenate all multihead context vector
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_k, -1)  # B X N X d
        # linear layer after attention
        attn_output = self.attn_linear(attn_output)
        attn_output = self.attn_output_dropout(attn_output)
        return attn_output, attns


if __name__ == "__main__":
    attention = Attention(8)
    X1 = torch.randn((5, 8))
    X2 = torch.randn((7, 8))
    mlp = MLP(8, [], 1)
    print(mlp(X1))
