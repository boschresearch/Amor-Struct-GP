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
from amorstructgp.config.kernels.kernel_list_configs import BasicKernelListConfig
from amorstructgp.config.models.base_model_config import BaseModelConfig
from amorstructgp.utils.enums import PredictionQuantity
from amorstructgp.config.nn.amortized_infer_models_configs import (
    BasicAmortizedInferenceModelConfig,
    BasicDimWiseAdditiveKernelAmortizedModelConfig,
    WiderCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
)
from amorstructgp.gp.base_symbols import BaseKernelTypes
from amorstructgp.utils.gaussian_mixture_density import EntropyApproximation


class BasicGPModelAmortizedEnsembleConfig(BaseModelConfig):
    kernel_list: List[List[BaseKernelTypes]]
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    entropy_approximation: EntropyApproximation = EntropyApproximation.MOMENT_MATCHED_GAUSSIAN
    amortized_model_config: BasicAmortizedInferenceModelConfig
    checkpoint_path: str


class ExperimentalAmortizedEnsembleConfig(BasicGPModelAmortizedEnsembleConfig):
    name: str = "ExperimentalAmortizedEnsemble"
    kernel_list: List[List[BaseKernelTypes]] = [
        [BaseKernelTypes.SE],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN],
        [BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE_MULT_LIN, BaseKernelTypes.LIN],
        [BaseKernelTypes.PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.LIN_MULT_PER, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN],
    ]
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )


class PaperAmortizedEnsembleConfig(BasicGPModelAmortizedEnsembleConfig):
    name: str = "PaperAmortizedEnsemble"
    kernel_list: List[List[BaseKernelTypes]] = [
        [BaseKernelTypes.SE],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN],
        [BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE_MULT_LIN, BaseKernelTypes.LIN],
        [BaseKernelTypes.PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.PER, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.LIN_MULT_PER, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN],
    ]  # this list is only the default list - not used in the paper - it can be changed after initialization
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )


class AmortizedEnsembleWithMaternConfig(BasicGPModelAmortizedEnsembleConfig):
    name: str = "AmortizedEnsembleWithMatern"
    kernel_list: List[List[BaseKernelTypes]] = [
        [BaseKernelTypes.SE],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN],
        [BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.SE_MULT_LIN],
        [BaseKernelTypes.SE_MULT_LIN, BaseKernelTypes.LIN],
        [BaseKernelTypes.MATERN52],
        [BaseKernelTypes.MATERN52, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN_MULT_MATERN52],
        [BaseKernelTypes.MATERN52, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_MATERN52],
        [BaseKernelTypes.LIN_MULT_MATERN52, BaseKernelTypes.LIN],
        [BaseKernelTypes.LIN],
    ]  # this list is only the default list - not used in the paper - it can be changed after initialization
    amortized_model_config: BasicAmortizedInferenceModelConfig = (
        SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    )
