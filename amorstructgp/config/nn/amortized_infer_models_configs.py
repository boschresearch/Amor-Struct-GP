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

from amorstructgp.config.nn.dataset_encoder_configs import (
    BaseDatasetEncoderConfig,
    DatasetEncoderConfig,
    SmallDatasetEncoderConfig,
    EnrichedDatasetEncoderConfig,
    SmallEnrichedDatasetEncoderConfig,
    FullDimDatasetEncoderConfig,
    SmallFullDimDatasetEncoderConfig,
    EnrichedStandardDatasetEncoderConfig,
)
from amorstructgp.config.nn.kernel_encoder_configs import (
    BaseKernelEncoderConfig,
    CrossAttentionKernelEncoderConfig,
    KernelEncoderConfig,
    KernelEncoderUnsharedWeightsAdditionConfig,
    SmallCrossAttentionKernelEncoderConfig,
    SmallKernelEncoderConfig,
    CrossAttentionKernelEncoderConfig,
    CrossAttentionMLPKernelEncoderConfig,
    CrossAttentionMLPStandardKernelEncoderConfig,
)
from amorstructgp.config.nn.noise_predictor_configs import (
    BaseNoisePredictorConfig,
    NoisePredictorConfig,
    NoisePredictorLargerLowerBoundConfig,
    NoisePredictorWithDatasetEncodingConfig,
    NoisePredictorWithDatasetEncodingLargerBoundConfig,
    NoisePredictorWithDatasetEncodingLargerBoundScaledConfig,
)


class BasicAmortizedInferenceModelConfig(BaseSettings):
    name: str
    dataset_encoder_config: BaseDatasetEncoderConfig
    kernel_encoder_config: BaseKernelEncoderConfig
    noise_predictor_config: BaseNoisePredictorConfig
    has_fix_noise_variance: bool
    gp_variance: float = 1e-2


class BasicDimWiseAdditiveKernelAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "BasicDimWiseAdditiveKernelAmortizedModel"
    dataset_encoder_config: BaseDatasetEncoderConfig = DatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = KernelEncoderConfig(dataset_enc_dim=dataset_encoder_config.output_embedding_dim)
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool


class DimWiseAdditiveKernelAmortizedModelConfig(BasicDimWiseAdditiveKernelAmortizedModelConfig):
    has_fix_noise_variance: bool = True
    name: str = "DimWiseAdditiveKernelAmortizedModel"


class DimWiseAdditiveKernelWithNoiseAmortizedModelConfig(BasicDimWiseAdditiveKernelAmortizedModelConfig):
    has_fix_noise_variance: bool = False
    name: str = "DimWiseAdditiveKernelWithNoiseAmortizedModel"


class SmallerDimWiseAdditiveKernelWithNoiseAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    dataset_encoder_config: BaseDatasetEncoderConfig = SmallDatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = SmallKernelEncoderConfig(dataset_enc_dim=dataset_encoder_config.output_embedding_dim)
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False
    name: str = "SmallerDimWiseAdditiveKernelWithNoiseAmortizedModel"


class DimWiseAdditiveWithNoiseUnsharedWeightsAdditionAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    dataset_encoder_config: BaseDatasetEncoderConfig = DatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = KernelEncoderUnsharedWeightsAdditionConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False
    name: str = "DimWiseAdditiveWithNoiseSharedWeightsAdditionAmortizedModel"


class SharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "SharedDatasetEncodingDimWiseAdditiveAmortizedModel"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = SmallKernelEncoderConfig(dataset_enc_dim=dataset_encoder_config.output_embedding_dim)
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorLargerLowerBoundConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class SmallerSharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "SmallerSharedDatasetEncodingDimWiseAdditiveAmortizedModel"
    dataset_encoder_config: BaseDatasetEncoderConfig = SmallEnrichedDatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = SmallKernelEncoderConfig(dataset_enc_dim=dataset_encoder_config.output_embedding_dim)
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorLargerLowerBoundConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class FullDimGlobalNoiseDimWiseAdditiveAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "FullDimGlobalNoiseDimWiseAdditiveAmortizedModel"
    dataset_encoder_config: BaseDatasetEncoderConfig = SmallFullDimDatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = SmallKernelEncoderConfig(dataset_enc_dim=dataset_encoder_config.output_embedding_dim)
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class CrossAttentionKernelEncSharedDatasetEncAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "CrossAttentionKernelEncSharedDatasetEncAmortizedModel"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = SmallCrossAttentionKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class BiggerCrossAttentionKernelEncSharedDatasetEncAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "BiggerCrossAttentionKernelEncSharedDatasetEncAmortizedModel"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig()
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class WiderCrossAttentionKernelEncSharedDatasetEncAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "WiderCrossAttentionKernelEncSharedDatasetEncAmortizedModel"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig(
        dataset_encoding_num_hidden_dim=512, output_embedding_dim=1024
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, hidden_dim=1024
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class ExperimentalAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """
    name: str = "ExperimentalAmortizedModel"
    dropout_p: float = 0.0
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedStandardDatasetEncoderConfig(
        dim_intermediate_bert=512, dataset_encoding_num_att_2=4, dataset_encoding_num_att_4=4, dropout_p=dropout_p
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPStandardKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, n_enc_layer=3, n_cross_layer=4, n_dec_layer=3, dropout_p=dropout_p
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
        dropout_p=dropout_p,
    )
    has_fix_noise_variance: bool = False


class ExperimentalAmortizedModelConfig2(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "ExperimentConfig"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig(
        dataset_encoding_num_hidden_dim=512,
        output_embedding_dim=1024,
        dataset_encoding_num_att_1=3,
        dataset_encoding_num_att_2=3,
        dataset_encoding_num_att_3=3,
        dataset_encoding_num_att_4=3,
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, hidden_dim=1024, n_enc_layer=3, n_cross_layer=3, n_dec_layer=3
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class WiderCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "WiderCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig(
        dataset_encoding_num_hidden_dim=512, output_embedding_dim=1024
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, hidden_dim=1024
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class WiderStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "WiderStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedStandardDatasetEncoderConfig(
        dataset_encoding_num_hidden_dim=512, output_embedding_dim=1024
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPStandardKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, hidden_dim=1024
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class SmallerStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "SmallerStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedStandardDatasetEncoderConfig(
        dim_intermediate_bert=512, dataset_encoding_num_att_2=4, dataset_encoding_num_att_4=4
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPStandardKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim,
        n_enc_layer=3,
        n_cross_layer=4,
        n_dec_layer=3,
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class ExperimentalCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "ExperimentalCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dropout_p: float = 0.05
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig(dropout_p=dropout_p)
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, dropout_p=dropout_p
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
        dropout_p=dropout_p,
    )
    has_fix_noise_variance: bool = False


class SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dropout_p: float = 0.0
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedStandardDatasetEncoderConfig(
        dim_intermediate_bert=512, dataset_encoding_num_att_2=4, dataset_encoding_num_att_4=4, dropout_p=dropout_p
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPStandardKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, n_enc_layer=3, n_cross_layer=4, n_dec_layer=3, dropout_p=dropout_p
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
        dropout_p=dropout_p,
    )
    has_fix_noise_variance: bool = False


class SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(
    BasicAmortizedInferenceModelConfig
):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dropout_p: float = 0.0
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedStandardDatasetEncoderConfig(
        dim_intermediate_bert=512, dataset_encoding_num_att_2=4, dataset_encoding_num_att_4=4, dropout_p=dropout_p
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPStandardKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, n_enc_layer=3, n_cross_layer=4, n_dec_layer=3, dropout_p=dropout_p
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
        dropout_p=dropout_p,
    )
    has_fix_noise_variance: bool = False


class ExperimentalSmallerStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "ExperimentalSmallerStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dropout_p: float = 0.025
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedStandardDatasetEncoderConfig(
        dim_intermediate_bert=512, dataset_encoding_num_att_2=4, dataset_encoding_num_att_4=4, dropout_p=dropout_p
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPStandardKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, n_enc_layer=3, n_cross_layer=4, n_dec_layer=3, dropout_p=dropout_p
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
        dropout_p=dropout_p,
    )
    has_fix_noise_variance: bool = False


class DeeperCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    """
    Config that is not persistent - for changing experiments - config gets logged anyway
    """

    name: str = "DeeperCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig"
    dataset_encoder_config: BaseDatasetEncoderConfig = EnrichedDatasetEncoderConfig(
        dataset_encoding_num_att_1=6, dataset_encoding_num_att_2=8, dataset_encoding_num_att_3=6, dataset_encoding_num_att_4=8
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPKernelEncoderConfig(
        dataset_enc_dim=dataset_encoder_config.output_embedding_dim, n_enc_layer=8, n_cross_layer=8, n_dec_layer=8
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorWithDatasetEncodingLargerBoundConfig(
        dim_hidden_layer_list=[200, 100],
        kernel_embedding_dim=kernel_encoder_config.hidden_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    has_fix_noise_variance: bool = False


class ARDRBFOnlyAmortizedModelConfig(BasicAmortizedInferenceModelConfig):
    name: str = "ARDRBFOnlyAmortizedModelg"
    dataset_encoder_config: BaseDatasetEncoderConfig = DatasetEncoderConfig(
        dataset_encoding_num_att_1=12,
        dataset_encoding_num_att_2=12,
        dataset_encoding_num_hidden_dim_1=512,
        dataset_encoding_num_hidden_dim_2=512,
        output_embedding_dim=512,
    )
    kernel_encoder_config: BaseKernelEncoderConfig = CrossAttentionMLPKernelEncoderConfig(
        dataset_enc_dim=0, n_enc_layer=0, n_cross_layer=0, n_dec_layer=0
    )
    noise_predictor_config: BaseNoisePredictorConfig = NoisePredictorLargerLowerBoundConfig(
        dim_hidden_layer_list=[1024, 512],
        kernel_embedding_dim=dataset_encoder_config.output_embedding_dim,
        dataset_encoding_dim=dataset_encoder_config.output_embedding_dim,
    )
    kernel_wrapper_hidden_layer_list = [1024, 512]
    kernel_wrapper_dropout_p = 0.1
    kernel_wrapper_hidden_dim = dataset_encoder_config.output_embedding_dim
    has_fix_noise_variance: bool = False


class BiggerARDRBFOnlyAmortizedModelConfig(ARDRBFOnlyAmortizedModelConfig):
    name: str = "BiggerARDRBFOnlyAmortizedModelg"
    dataset_encoder_config: BaseDatasetEncoderConfig = DatasetEncoderConfig(
        dataset_encoding_num_att_1=14,
        dataset_encoding_num_att_2=14,
        dataset_encoding_num_hidden_dim_1=512,
        dataset_encoding_num_hidden_dim_2=512,
        output_embedding_dim=512,
    )


if __name__ == "__main__":
    config = SharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig()
    print(config.dict())
