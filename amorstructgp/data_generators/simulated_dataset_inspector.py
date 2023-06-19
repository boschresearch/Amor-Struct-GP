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
from typing import Type
from amorstructgp.data_generators.base_data_generator import BaseDataGenerator
from amorstructgp.data_generators.dataset_of_datasets import DatasetType
from amorstructgp.data_generators.dim_wise_additive_kernels_generator import (
    DimWiseAdditiveKernelGenerator,
)
from amorstructgp.data_generators.simulator import load_simulated_dataset_from_files
from amorstructgp.utils.utils import read_dict_from_json, read_list_from_file
import os
import numpy as np


class SimulatedDatasetInspector:
    def __init__(self, generator_class: Type[BaseDataGenerator], main_folder: str, dataset_type: DatasetType):
        self.main_folder = main_folder
        train_file_name, val_file_name, test_file_name = generator_class.get_main_file_names()
        train_dimension_index_file, val_dimension_index_file, test_dimension_index_file = generator_class.get_dimension_index_file_names()
        if dataset_type == DatasetType.TRAIN:
            self.dimension_dict = read_dict_from_json(os.path.join(main_folder, train_dimension_index_file))
        elif dataset_type == DatasetType.VAL:
            self.dimension_dict = read_dict_from_json(os.path.join(main_folder, val_dimension_index_file))
        elif dataset_type == DatasetType.TEST:
            self.dimension_dict = read_dict_from_json(os.path.join(main_folder, test_dimension_index_file))

    def inspect_dimension(self, n: int, dimension: int):
        uuid_list = self.dimension_dict[str(dimension)]
        if n > len(uuid_list):
            n = len(uuid_list)

        random_uuids = list(np.random.choice(uuid_list, n, replace=False))
        for uuid in random_uuids:
            simulated_dataset_object = load_simulated_dataset_from_files(self.main_folder, uuid)
            print(simulated_dataset_object)
            simulated_dataset_object.plot()


if __name__ == "__main__":
    save_folder = "C:\\Users\\BIM2RNG\\Desktop\\workspace\\temp\\amort_infer_data\\test3"
    inspector = SimulatedDatasetInspector(DimWiseAdditiveKernelGenerator, save_folder, DatasetType.TRAIN)
    inspector.inspect_dimension(30, 2)
