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
from enum import IntEnum
from typing import List, Tuple, Union


class BaseKernelTypes(IntEnum):
    SE = 0
    LIN = 1
    PER = 2
    MATERN52 = 3
    SE_MULT_LIN = 4
    SE_MULT_PER = 5
    SE_MULT_MATERN52 = 6
    LIN_MULT_PER = 7
    LIN_MULT_MATERN52 = 8
    PER_MULT_MATERN52 = 9


N_BASE_KERNEL_TYPES = 10
