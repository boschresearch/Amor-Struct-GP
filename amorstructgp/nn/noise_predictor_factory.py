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
from amorstructgp.config.nn.noise_predictor_configs import (
    BaseNoisePredictorConfig,
    NoisePredictorConfig,
    NoisePredictorWithDatasetEncodingConfig,
)
from amorstructgp.nn.noise_predictor_head import (
    NoiseVariancePredictorHead,
    NoiseVariancePredictorHeadWithDatasetEncoding,
)


class NoisePredictorFactory:
    @staticmethod
    def build(noise_predictor_config: BaseNoisePredictorConfig):
        if isinstance(noise_predictor_config, NoisePredictorConfig):
            noise_variance_predictor = NoiseVariancePredictorHead(**noise_predictor_config.dict())
            return noise_variance_predictor
        elif isinstance(noise_predictor_config, NoisePredictorWithDatasetEncodingConfig):
            noise_variance_predictor = NoiseVariancePredictorHeadWithDatasetEncoding(**noise_predictor_config.dict())
            return noise_variance_predictor
        else:
            raise ValueError
