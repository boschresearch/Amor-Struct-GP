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
from amorstructgp.config.data_generators.dim_wise_additive_generator_config import (
    BasicDimWiseAdditiveGeneratorConfig,
    DimWiseAdditiveWithNoiseMixedConfig,
)
from amorstructgp.data_generators.dim_wise_additive_kernels_generator import (
    DimWiseAdditiveKernelGenerator,
)


class GeneratorFactory:
    @staticmethod
    def build(generator_config):
        if isinstance(generator_config, BasicDimWiseAdditiveGeneratorConfig):
            generator = DimWiseAdditiveKernelGenerator(**generator_config.dict())
            return generator
        else:
            raise ValueError()


if __name__ == "__main__":
    factory = GeneratorFactory()
    generator = GeneratorFactory.build(DimWiseAdditiveWithNoiseMixedConfig())
