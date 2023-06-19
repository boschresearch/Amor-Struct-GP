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


class BaseNoisePredictorConfig(BaseSettings):
    name: str
    kernel_embedding_dim: int
    dataset_encoding_dim: int
    use_scaled_softplus: bool = False


class NoisePredictorConfig(BaseNoisePredictorConfig):
    name: str = "NoisePredictor"
    dim_hidden_layer_list: List[int] = [40]
    noise_variance_lower_bound: float = 1e-4
    dropout_p: float = 0.2


class NoisePredictorLargerLowerBoundConfig(NoisePredictorConfig):
    name: str = "NoisePredictorLargerLowerBound"
    noise_variance_lower_bound: float = 4e-4


class NoisePredictorWithDatasetEncodingConfig(BaseNoisePredictorConfig):
    name: str = "NoisePredictorWithDatasetEncoding"
    dim_hidden_layer_list: List[int] = [40]
    dropout_p: float = 0.2
    noise_variance_lower_bound: float = 1e-4


class NoisePredictorWithDatasetEncodingLargerBoundConfig(NoisePredictorWithDatasetEncodingConfig):
    name: str = "NoisePredictorWithDatasetEncodingLargerBound"
    noise_variance_lower_bound: float = 4e-4


class NoisePredictorWithDatasetEncodingLargerBoundScaledConfig(NoisePredictorWithDatasetEncodingLargerBoundConfig):
    name: str = "NoisePredictorWithDatasetEncodingLargerBoundScaled"
    use_scaled_softplus: bool = True
