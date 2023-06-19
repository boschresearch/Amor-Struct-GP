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
from amorstructgp.config.nn.dataset_encoder_configs import (
    BaseDatasetEncoderConfig,
    DatasetEncoderConfig,
    EnrichedDatasetEncoderConfig,
    FullDimDatasetEncoderConfig,
)
from amorstructgp.nn.dataset_encoder import (
    DatasetEncoder,
    EnrichedDatasetEncoder,
    FullDimDatasetEncoder,
)


class DatasetEncoderFactory:
    @staticmethod
    def build(dataset_encoder_config: BaseDatasetEncoderConfig):
        if isinstance(dataset_encoder_config, DatasetEncoderConfig):
            dataset_encoder = DatasetEncoder(
                dataset_encoder_config.dataset_encoding_num_att_1,
                dataset_encoder_config.dataset_encoding_num_att_2,
                dataset_encoder_config.dataset_encoding_num_hidden_dim_1,
                dataset_encoder_config.dataset_encoding_num_hidden_dim_2,
                dataset_encoder_config.dropout_p,
            )
            return dataset_encoder
        elif isinstance(dataset_encoder_config, EnrichedDatasetEncoderConfig):
            dataset_encoder = EnrichedDatasetEncoder(
                dataset_encoder_config.dataset_encoding_num_att_1,
                dataset_encoder_config.dataset_encoding_num_att_2,
                dataset_encoder_config.dataset_encoding_num_att_3,
                dataset_encoder_config.dataset_encoding_num_att_4,
                dataset_encoder_config.dataset_encoding_num_hidden_dim,
                dataset_encoder_config.use_standard_layers,
                dataset_encoder_config.dim_intermediate_bert,
                dataset_encoder_config.dropout_p,
            )
            return dataset_encoder
        elif isinstance(dataset_encoder_config, FullDimDatasetEncoderConfig):
            dataset_encoder = FullDimDatasetEncoder(
                dataset_encoder_config.dataset_encoding_num_att_1,
                dataset_encoder_config.dataset_encoding_num_att_2,
                dataset_encoder_config.dataset_encoding_num_hidden_dim_1,
                dataset_encoder_config.dataset_encoding_num_hidden_dim_2,
                dataset_encoder_config.max_dim,
                dataset_encoder_config.dropout_p,
            )
            return dataset_encoder
        else:
            raise ValueError
