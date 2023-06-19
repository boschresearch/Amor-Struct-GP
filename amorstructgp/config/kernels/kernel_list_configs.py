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
from amorstructgp.config.kernels.base_kernel_config import BaseKernelConfig
from amorstructgp.gp.base_symbols import BaseKernelTypes


class BasicKernelListConfig(BaseKernelConfig):
    kernel_list: List[BaseKernelTypes]
    add_prior: bool = False


class SEKernelViaKernelListConfig(BasicKernelListConfig):
    name: str = "SEKernelViaKernelList"
    kernel_list: List[BaseKernelTypes] = [BaseKernelTypes.SE]


class PERKernelViaKernelListConfig(BasicKernelListConfig):
    name: str = "PERKernelViaKernelList"
    kernel_list: List[BaseKernelTypes] = [BaseKernelTypes.PER]


class MaternKernelViaKernelListConfig(BasicKernelListConfig):
    name: str = "MaternKernelViaKernelList"
    kernel_list: List[BaseKernelTypes] = [BaseKernelTypes.MATERN52]


class ExperimentalKernelListConfig(BaseKernelConfig):
    name: str = "ExperimentalKernelList"
    kernel_list: List[BaseKernelTypes] = [BaseKernelTypes.SE]
