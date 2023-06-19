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
from pydantic import BaseSettings


class BaseKernelEncoderConfig(BaseSettings):
    name: str
    input_dim: int = 10
    hidden_dim: int
    dataset_enc_dim: int
    share_weights_in_additions: bool
    kernel_wrapper_hidden_layer_list: List[int] = []
    dropout_p: float = 0.1


class KernelEncoderConfig(BaseKernelEncoderConfig):
    name: str = "KernelEncoder"
    n_enc_dec_layer: int = 8
    hidden_dim: int = 512
    share_weights_in_additions: bool = True


class KernelEncoderUnsharedWeightsAdditionConfig(KernelEncoderConfig):
    name: str = "KernelEncoderSharedWeightsAddition"
    share_weights_in_additions: bool = False


class SmallKernelEncoderConfig(KernelEncoderConfig):
    name: str = "SmallKernelEncoder"
    hidden_dim: int = 256


class CrossAttentionKernelEncoderConfig(BaseKernelEncoderConfig):
    name: str = "CrossAttentionKernelEncoder"
    n_enc_layer: int = 6
    n_cross_layer: int = 6
    n_dec_layer: int = 6
    hidden_dim: int = 512
    share_weights_in_additions: bool = True
    use_standard_layers: bool = False


class CrossAttentionMLPKernelEncoderConfig(CrossAttentionKernelEncoderConfig):
    name: str = "CrossAttentionMLPKernelEncoder"
    kernel_wrapper_hidden_layer_list: List[int] = [200]


class CrossAttentionMLPStandardKernelEncoderConfig(CrossAttentionKernelEncoderConfig):
    name: str = "CrossAttentionMLPKernelEncoder"
    kernel_wrapper_hidden_layer_list: List[int] = [200]
    use_standard_layers: bool = True


class SmallCrossAttentionKernelEncoderConfig(CrossAttentionKernelEncoderConfig):
    name: str = "SmallCrossAttentionKernelEncoder"
    hidden_dim: int = 256
