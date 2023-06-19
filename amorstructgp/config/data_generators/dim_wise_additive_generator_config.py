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

from typing import Tuple
from pydantic import BaseSettings


class BasicDimWiseAdditiveGeneratorConfig(BaseSettings):
    min_max_n: Tuple[int, int]
    min_max_d: Tuple[int, int]
    kernel_len_geometric_p: float
    uniform_kernel_selection: bool
    n_test: int
    replace_in_kernel_list: bool
    observation_noise: float
    sample_observation_noise: bool
    add_negative_kernels_train: bool
    fraction_gt_kernels_in_train: float
    include_matern: bool
    num_train_datasets_on_the_fly: int
    num_val_datasets_on_the_fly: int
    num_test_datasets_on_the_fly: int


class DimWiseAdditiveWithNoiseMixedConfig(BasicDimWiseAdditiveGeneratorConfig):
    name = "DimWiseAdditiveWithNoiseMixed"
    min_max_n: Tuple[int, int] = (10, 150)
    min_max_d: Tuple[int, int] = (1, 4)
    kernel_len_geometric_p: float = 0.6
    num_dimension_geometric_p: float = 0.3
    uniform_dist_for_dimension: bool = True
    uniform_kernel_selection: bool = False
    n_test: int = 50
    replace_in_kernel_list: bool = False
    observation_noise: float = 0.1
    sample_observation_noise: bool = True
    add_negative_kernels_train: bool = True
    fraction_gt_kernels_in_train: float = 0.5
    include_matern: bool = True
    num_train_datasets_on_the_fly: int = 40000
    num_val_datasets_on_the_fly: int = 2000
    num_test_datasets_on_the_fly: int = 2000


class DimWiseAdditiveWithNoiseMixedNoMaternConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "DimWiseAdditiveWithNoiseMixedNoMatern"
    include_matern: bool = False


class DimWiseAdditiveWithNoiseMixedNoMaternNoNegativeConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "DimWiseAdditiveWithNoiseMixedNoMaternNoNegative"
    include_matern: bool = False
    add_negative_kernels_train: bool = False


class DimWiseAdditiveWithNoiseMixedNoMaternBiggerConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "BaseDimWiseAdditiveWithNoiseMixedNoMaternBigger"
    min_max_n: Tuple[int, int] = (10, 250)
    min_max_d: Tuple[int, int] = (1, 8)
    num_dimension_geometric_p: float = 0.25
    uniform_dist_for_dimension: bool = False
    include_matern: bool = False


class DimWiseAdditiveWithNoiseMixedWithMaternBiggerConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "BaseDimWiseAdditiveWithNoiseMixedWithMaternBigger"
    min_max_n: Tuple[int, int] = (10, 250)
    min_max_d: Tuple[int, int] = (1, 8)
    num_dimension_geometric_p: float = 0.25
    uniform_dist_for_dimension: bool = False
    include_matern: bool = True

class DimWiseAdditiveWithNoiseMixedWithMaternBiggerOnlyPositiveConfig(DimWiseAdditiveWithNoiseMixedWithMaternBiggerConfig):
    name = "DimWiseAdditiveWithNoiseMixedWithMaternBiggerOnlyPositive"
    add_negative_kernels_train : bool = False

class OneDTimeSeriesAdditiveWithNoiseMixedNoMaternBiggerConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "OneDTimeSeriesAdditiveWithNoiseMixedNoMaternBiggerConfig"
    min_max_n: Tuple[int, int] = (10, 250)
    min_max_d: Tuple[int, int] = (1, 1)
    uniform_dist_for_dimension: bool = True
    include_matern: bool = False
    fraction_gt_kernels_in_train: float = 0.8


class OneDTimeSeriesAdditiveWithNoiseOnlyPositiveNoMaternBiggerConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "OneDTimeSeriesAdditiveWithNoiseOnlyPositiveNoMaternBiggerConfig"
    min_max_n: Tuple[int, int] = (150, 250)
    min_max_d: Tuple[int, int] = (1, 1)
    uniform_dist_for_dimension: bool = True
    include_matern: bool = False
    add_negative_kernels_train: bool = False


class DimWiseAdditiveWithNoiseMixedNoMaternBiggerOnlyPositiveConfig(DimWiseAdditiveWithNoiseMixedNoMaternBiggerConfig):
    name = "BaseDimWiseAdditiveWithNoiseMixedNoMaternBiggerOnlyPossitive"
    add_negative_kernels_train: bool = False


class DimWiseAdditiveWithNoiseMixedNoMaternMoreGTConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "DimWiseAdditiveWithNoiseMixedNoMaternMoreGTConfig"
    min_max_n: Tuple[int, int] = (10, 250)
    min_max_d: Tuple[int, int] = (1, 8)
    fraction_gt_kernels_in_train: float = 0.8
    num_dimension_geometric_p: float = 0.25
    uniform_dist_for_dimension: bool = False
    include_matern: bool = False


class DimWiseAdditiveWithNoiseMixedNoMaternVeryLargeConfig(DimWiseAdditiveWithNoiseMixedConfig):
    name = "DimWiseAdditiveWithNoiseMixedNoMaternVeryLarge"
    min_max_n: Tuple[int, int] = (10, 500)
    min_max_d: Tuple[int, int] = (1, 8)
    num_dimension_geometric_p: float = 0.25
    uniform_dist_for_dimension: bool = False
    include_matern: bool = False


if __name__ == "__main__":
    config = DimWiseAdditiveWithNoiseMixedNoMaternBiggerConfig()
    print(config.dict())
