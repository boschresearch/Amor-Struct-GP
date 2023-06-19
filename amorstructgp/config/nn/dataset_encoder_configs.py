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
from pydantic import BaseSettings


class BaseDatasetEncoderConfig(BaseSettings):
    name: str
    output_embedding_dim: int
    dropout_p: float = 0.1


class DatasetEncoderConfig(BaseDatasetEncoderConfig):
    name: str = "DatasetEncoder"
    dataset_encoding_num_att_1: int = 8
    dataset_encoding_num_att_2: int = 8
    dataset_encoding_num_hidden_dim_1: int = 512
    dataset_encoding_num_hidden_dim_2: int = 512
    output_embedding_dim: int = 512


class SmallDatasetEncoderConfig(DatasetEncoderConfig):
    name: str = "SmallDatasetEncoder"
    dataset_encoding_num_hidden_dim_1: int = 256
    dataset_encoding_num_hidden_dim_2: int = 256
    output_embedding_dim: int = 256


class EnrichedDatasetEncoderConfig(BaseDatasetEncoderConfig):
    name: str = "EnrichedDatasetEncoder"
    dataset_encoding_num_att_1: int = 4
    dataset_encoding_num_att_2: int = 6
    dataset_encoding_num_att_3: int = 4
    dataset_encoding_num_att_4: int = 6
    dataset_encoding_num_hidden_dim: int = 256
    output_embedding_dim: int = 512
    dim_intermediate_bert: int = 1024
    use_standard_layers: bool = False


class EnrichedStandardDatasetEncoderConfig(EnrichedDatasetEncoderConfig):
    name: str = "EnrichedStandardDatasetEncoder"
    use_standard_layers: bool = True


class SmallEnrichedDatasetEncoderConfig(EnrichedDatasetEncoderConfig):
    name: str = "SmallEnrichedDatasetEncoder"
    dataset_encoding_num_hidden_dim: int = 128
    output_embedding_dim: int = 256


class FullDimDatasetEncoderConfig(BaseDatasetEncoderConfig):
    name: str = "FullDimDatasetEncoder"
    max_dim: int = 10
    dataset_encoding_num_att_1: int = 8
    dataset_encoding_num_att_2: int = 8
    dataset_encoding_num_hidden_dim_1: int = 512
    dataset_encoding_num_hidden_dim_2: int = 512
    output_embedding_dim: int = 512


class SmallFullDimDatasetEncoderConfig(FullDimDatasetEncoderConfig):
    name: str = "SmallFullDimDatasetEncoder"
    dataset_encoding_num_hidden_dim_1: int = 256
    dataset_encoding_num_hidden_dim_2: int = 256
    output_embedding_dim: int = 256
