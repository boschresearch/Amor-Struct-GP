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
from amorstructgp.config.nn.amortized_infer_models_configs import (
    ARDRBFOnlyAmortizedModelConfig,
    BasicAmortizedInferenceModelConfig,
    BasicDimWiseAdditiveKernelAmortizedModelConfig,
    DimWiseAdditiveKernelAmortizedModelConfig,
    DimWiseAdditiveKernelWithNoiseAmortizedModelConfig,
    SharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig,
)
from amorstructgp.nn.dataset_encoder import DatasetEncoder, EnrichedDatasetEncoder
from amorstructgp.nn.dataset_encoder_factory import DatasetEncoderFactory
from amorstructgp.nn.kernel_encoder import KernelEncoder
from amorstructgp.nn.amortized_inference_models import (
    ARDRBFOnlyAmortizedInferenceModel,
    DimWiseAdditiveKernelsAmortizedInferenceModel,
    DimWiseAdditiveKernelWithNoiseAmortizedModel,
)
from amorstructgp.nn.kernel_encoder_factory import KernelEncoderFactory
from amorstructgp.nn.noise_predictor_factory import NoisePredictorFactory
from amorstructgp.nn.noise_predictor_head import NoiseVariancePredictorHead


class AmortizedModelsFactory:
    @staticmethod
    def build(model_config: BasicAmortizedInferenceModelConfig):
        if isinstance(model_config, ARDRBFOnlyAmortizedModelConfig):
            dataset_encoder = DatasetEncoderFactory.build(model_config.dataset_encoder_config)
            noise_variance_predictor = NoisePredictorFactory.build(model_config.noise_predictor_config)
            model = ARDRBFOnlyAmortizedInferenceModel(
                dataset_encoder,
                noise_variance_predictor,
                model_config.kernel_wrapper_hidden_layer_list,
                model_config.kernel_wrapper_dropout_p,
                model_config.kernel_wrapper_hidden_dim,
            )
            return model
        dataset_encoder = DatasetEncoderFactory.build(model_config.dataset_encoder_config)
        kernel_encoder = KernelEncoderFactory.build(model_config.kernel_encoder_config)
        if model_config.has_fix_noise_variance:
            model = DimWiseAdditiveKernelsAmortizedInferenceModel(
                dataset_encoder,
                kernel_encoder,
                model_config.kernel_encoder_config.share_weights_in_additions,
                model_config.kernel_encoder_config.kernel_wrapper_hidden_layer_list,
                model_config.kernel_encoder_config.dropout_p,
                model_config.gp_variance,
            )
        else:
            noise_variance_predictor = NoisePredictorFactory.build(model_config.noise_predictor_config)
            model = DimWiseAdditiveKernelWithNoiseAmortizedModel(
                dataset_encoder,
                kernel_encoder,
                noise_variance_predictor,
                model_config.kernel_encoder_config.share_weights_in_additions,
                model_config.kernel_encoder_config.kernel_wrapper_hidden_layer_list,
                model_config.kernel_encoder_config.dropout_p,
            )
        return model
