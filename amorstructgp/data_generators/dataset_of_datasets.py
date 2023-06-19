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
import os
from typing import List, Type
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from amorstructgp.data_generators.base_data_generator import BaseDataGenerator
from amorstructgp.data_generators.dim_wise_additive_kernels_generator import (
    DimWiseAdditiveKernelGenerator,
)
from amorstructgp.data_generators.warper import Warper
from amorstructgp.gp.base_kernels import get_kernel_embeddings_and_kernel_mask, flatten_nested_list
from amorstructgp.nn.dataset_encoder import get_padded_dataset_and_masks
from amorstructgp.utils.utils import read_dict_from_json, read_list_from_file
from amorstructgp.data_generators.simulator import (
    SimulatedDataset,
    load_simulated_dataset_from_files,
)
import numpy as np


class DatasetType(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


class DatasetOfDatasets(Dataset):
    def __init__(
        self,
        generator_class: Type[BaseDataGenerator],
        main_folder: str,
        dataset_type: DatasetType,
        normalize_datasets: bool,
        warp_inputs: bool,
        return_kernel_parameter_lists: bool,
    ) -> None:
        super().__init__()
        assert issubclass(generator_class, BaseDataGenerator)
        self.generator_class = generator_class
        self.main_folder = main_folder
        self.normalize_datasets = normalize_datasets
        self.return_kernel_parameter_lists = return_kernel_parameter_lists
        self.warp_inputs = warp_inputs
        train_file_name, val_file_name, test_file_name = generator_class.get_main_file_names()
        train_dimension_index_file, val_dimension_index_file, test_dimension_index_file = generator_class.get_dimension_index_file_names()
        if dataset_type == DatasetType.TRAIN:
            self.uuid_list = read_list_from_file(os.path.join(main_folder, train_file_name))
            # Dict that maps input_dimension-> List[uuids]
            self.dimension_dict = read_dict_from_json(os.path.join(main_folder, train_dimension_index_file))
        elif dataset_type == DatasetType.VAL:
            self.uuid_list = read_list_from_file(os.path.join(main_folder, val_file_name))
            self.dimension_dict = read_dict_from_json(os.path.join(main_folder, val_dimension_index_file))
        elif dataset_type == DatasetType.TEST:
            self.uuid_list = read_list_from_file(os.path.join(main_folder, test_file_name))
            self.dimension_dict = read_dict_from_json(os.path.join(main_folder, test_dimension_index_file))

    def __getitem__(self, index):
        simulated_dataset_object = load_simulated_dataset_from_files(self.main_folder, self.uuid_list[index])
        return simulated_dataset_object

    def __len__(self):
        return len(self.uuid_list)

    def collate_fn(self, batch):
        if issubclass(self.generator_class, DimWiseAdditiveKernelGenerator):
            return collate_fn_dim_wise_additive_kernel_gen(
                batch, self.normalize_datasets, self.warp_inputs, self.return_kernel_parameter_lists
            )

    def get_max_dimension(self):
        return max([int(dimension) for dimension in self.dimension_dict])

    def get_n_datasets_with_input_dimension(self, n: int, input_dimension: int) -> List[SimulatedDataset]:
        uuid_list = self.dimension_dict[str(input_dimension)]
        if n > len(uuid_list):
            n = len(uuid_list)
        random_uuids = list(np.random.choice(uuid_list, n, replace=False))
        list_of_datasets = []
        for uuid in random_uuids:
            simulated_dataset_object = load_simulated_dataset_from_files(self.main_folder, uuid)
            list_of_datasets.append(simulated_dataset_object)
        return list_of_datasets


class RandomDatasetOfDatasets(Dataset):
    def __init__(
        self,
        generator: BaseDataGenerator,
        normalize_datasets: bool,
        n_datasets: int,
        warp_inputs: bool,
        return_kernel_parameter_lists: bool,
    ) -> None:
        super().__init__()
        assert isinstance(generator, BaseDataGenerator)
        self.generator = generator
        self.normalize_datasets = normalize_datasets
        self.warp_inputs = warp_inputs
        self.return_kernel_parameter_lists = return_kernel_parameter_lists
        self.n_datasets = n_datasets

    def __getitem__(self, index):
        simulated_dataset_object = self.generator.generate_one_sample()
        return simulated_dataset_object

    def __len__(self):
        return self.n_datasets

    def collate_fn(self, batch):
        if isinstance(self.generator, DimWiseAdditiveKernelGenerator):
            return collate_fn_dim_wise_additive_kernel_gen(
                batch, self.normalize_datasets, self.warp_inputs, self.return_kernel_parameter_lists
            )

    def get_max_dimension(self):
        return self.generator.get_max_dimension()

    def get_n_datasets_with_input_dimension(self, n: int, input_dimension: int) -> List[SimulatedDataset]:
        list_of_datasets = []
        for i in range(0, n):
            simulated_dataset_object = self.generator.generate_one_sample(input_dimension)
            list_of_datasets.append(simulated_dataset_object)
        return list_of_datasets


######### Generator specfic collating functions #######################

######### DimWiseAdditiveKernelGenerator #############


def collate_fn_dim_wise_additive_kernel_gen(batch, normalize_datasets: bool, warp_inputs: bool, return_kernel_parameter_lists: bool):
    warper = Warper()
    X_list = []
    y_list = []
    input_kernel_list = []
    gt_kernel_parameter_lists = []
    mlls_gt = []
    log_posterior_density_gt = []
    observation_noise_gt = []
    for simulated_dataset in batch:
        assert isinstance(simulated_dataset, SimulatedDataset)
        x_data = simulated_dataset.X_data
        if warp_inputs:
            x_data = warper.apply(x_data)
        X_list.append(x_data)
        if normalize_datasets:
            y_list.append(np.squeeze(simulated_dataset.y_data_normalized))
        else:
            y_list.append(np.squeeze(simulated_dataset.y_data))
        input_kernel_list.append(simulated_dataset.get_input_kernel_list())
        mlls_gt.append(simulated_dataset.mll_gt)
        observation_noise_gt.append(simulated_dataset.get_observation_noise())
        log_posterior_density_gt.append(simulated_dataset.log_posterior_density_gt)
        if return_kernel_parameter_lists:
            kernel_parameter_list = simulated_dataset.get_gt_kernel_parameter_list()
            gt_kernel_parameter_lists.append(kernel_parameter_list)
    kernel_embeddings, kernel_mask = get_kernel_embeddings_and_kernel_mask(input_kernel_list)
    X_padded, y_padded, size_mask, dim_mask, N, size_mask_kernel = get_padded_dataset_and_masks(X_list, y_list)
    kernel_embeddings = torch.from_numpy(kernel_embeddings).float()
    kernel_mask = torch.from_numpy(kernel_mask).float()
    X_padded = torch.from_numpy(X_padded).float()
    y_padded = torch.from_numpy(y_padded).float()
    size_mask = torch.from_numpy(size_mask).float()
    dim_mask = torch.from_numpy(dim_mask).float()
    size_mask_kernel = torch.from_numpy(size_mask_kernel).float()
    N = torch.from_numpy(N).float()
    mlls_gt = torch.from_numpy(np.array(mlls_gt)).float()
    observation_noise_gt = torch.from_numpy(np.array(observation_noise_gt)).float()
    log_posterior_density_gt = torch.from_numpy(np.array(log_posterior_density_gt)).float()
    return (
        kernel_embeddings,
        kernel_mask,
        X_padded,
        y_padded,
        size_mask,
        dim_mask,
        size_mask_kernel,
        N,
        input_kernel_list,
        mlls_gt,
        log_posterior_density_gt,
        gt_kernel_parameter_lists,
        observation_noise_gt,
    )


if __name__ == "__main__":
    pass
