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

from amorstructgp.utils.enums import OptimizerType, LossFunctionType


class BasicDimWiseAdditiveKernelTrainingConfig(BaseSettings):
    num_epochs: int = 4500
    learning_rate: float = 1e-5
    batch_size: int = 32
    normalize_datasets: bool = False
    warp_inputs: bool = False
    use_lr_scheduler: bool = False
    freeze_dataset_encoder: bool = False
    optimizer_type: OptimizerType = OptimizerType.RADAM
    loss_function_type: LossFunctionType = LossFunctionType.NMLL


class DimWiseAdditiveKernelFineTuningConfig(BaseSettings):
    num_epochs: int = 100
    learning_rate: float = 1e-5
    batch_size: int = 32
    normalize_datasets: bool = False
    warp_inputs: bool = False
    use_lr_scheduler: bool = False
    freeze_dataset_encoder: bool = False
    optimizer_type: OptimizerType = OptimizerType.RADAM
    loss_function_type: LossFunctionType = LossFunctionType.NOISE_RMSE_PLUS_NMLL
