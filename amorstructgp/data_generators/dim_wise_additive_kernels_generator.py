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
import random
from typing import Dict, List, Optional, Tuple

import torch
from amorstructgp.data_generators.base_data_generator import BaseDataGenerator
from amorstructgp.data_generators.simulator import SimulatedDataset, Simulator

from amorstructgp.gp.base_kernels import BaseKernelTypes, transform_kernel_list_to_expression
import numpy as np
from amorstructgp.utils.utils import convert_list_elements_to_int

from amorstructgp.utils.utils import write_dict_to_json, write_list_to_file
from amorstructgp.config.prior_parameters import NOISE_VARIANCE_EXPONENTIAL_LAMBDA


class DimWiseAdditiveKernelGenerator(BaseDataGenerator):
    train_uuids_file = "train_uuids.txt"
    val_uuids_file = "val_uuids.txt"
    test_uuids_file = "test_uuids.txt"
    dimension_index_file_train = "train_dimension_index.json"
    dimension_index_file_val = "val_dimension_index.json"
    dimension_index_file_test = "test_dimension_index.json"

    @staticmethod
    def get_main_file_names() -> Tuple[str, str, str]:
        return (
            DimWiseAdditiveKernelGenerator.train_uuids_file,
            DimWiseAdditiveKernelGenerator.val_uuids_file,
            DimWiseAdditiveKernelGenerator.test_uuids_file,
        )

    @staticmethod
    def get_dimension_index_file_names() -> Tuple[str, str, str]:
        return (
            DimWiseAdditiveKernelGenerator.dimension_index_file_train,
            DimWiseAdditiveKernelGenerator.dimension_index_file_val,
            DimWiseAdditiveKernelGenerator.dimension_index_file_test,
        )

    def __init__(
        self,
        min_max_n: Tuple[int, int],
        min_max_d: Tuple[int, int],
        kernel_len_geometric_p: float,
        num_dimension_geometric_p: float,
        uniform_dist_for_dimension: bool,
        uniform_kernel_selection: bool,
        n_test: int,
        replace_in_kernel_list: bool,
        observation_noise: float,
        sample_observation_noise: bool,
        add_negative_kernels_train: bool,
        fraction_gt_kernels_in_train: float,
        include_matern: bool,
        **kwargs
    ):
        if include_matern:
            self.base_kerne_type_list = [
                BaseKernelTypes.SE,
                BaseKernelTypes.LIN,
                BaseKernelTypes.PER,
                BaseKernelTypes.MATERN52,
                BaseKernelTypes.SE_MULT_LIN,
                BaseKernelTypes.SE_MULT_PER,
                BaseKernelTypes.LIN_MULT_PER,
                BaseKernelTypes.SE_MULT_MATERN52,
                BaseKernelTypes.LIN_MULT_MATERN52,
                BaseKernelTypes.PER_MULT_MATERN52,
            ]
            if uniform_kernel_selection:
                self.unnormalized_weights_base_kernels = np.ones(10)
            else:
                self.unnormalized_weights_base_kernels = np.array([2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        else:
            self.base_kerne_type_list = [
                BaseKernelTypes.SE,
                BaseKernelTypes.LIN,
                BaseKernelTypes.PER,
                BaseKernelTypes.SE_MULT_LIN,
                BaseKernelTypes.SE_MULT_PER,
                BaseKernelTypes.LIN_MULT_PER,
            ]
            if uniform_kernel_selection:
                self.unnormalized_weights_base_kernels = np.ones(6)
            else:
                self.unnormalized_weights_base_kernels = np.array([2, 2, 2, 1, 1, 1], dtype=np.float32)

        self.kernel_len_geometric_p = kernel_len_geometric_p
        self.num_dimension_geometric_p = num_dimension_geometric_p
        self.uniform_dist_for_dimension = uniform_dist_for_dimension
        self.weights_base_kernels = self.unnormalized_weights_base_kernels / np.sum(self.unnormalized_weights_base_kernels)
        self.replace_in_kernel_list = replace_in_kernel_list
        self.min_max_n = min_max_n
        self.min_max_d = min_max_d
        self.n_test = n_test
        self.observation_noise = observation_noise
        self.sample_observation_noise = sample_observation_noise
        self.add_negative_kernels_train = add_negative_kernels_train
        self.fraction_gt_kernels_in_train = fraction_gt_kernels_in_train
        self.simulator = Simulator(0.0, 1.0, NOISE_VARIANCE_EXPONENTIAL_LAMBDA)

    def get_max_dimension(self):
        return self.min_max_d[1]

    def generate_simulated_data(self, n_train: int, n_val: int, n_test: int, folder: str):
        ids_train, dimension_index_train = self._generate_simulated_data(n_train, folder)
        ids_val, dimension_index_val = self._generate_simulated_data(n_val, folder)
        ids_test, dimension_index_test = self._generate_simulated_data(n_test, folder)
        write_list_to_file(ids_train, os.path.join(folder, DimWiseAdditiveKernelGenerator.train_uuids_file))
        write_list_to_file(ids_val, os.path.join(folder, DimWiseAdditiveKernelGenerator.val_uuids_file))
        write_list_to_file(ids_test, os.path.join(folder, DimWiseAdditiveKernelGenerator.test_uuids_file))
        write_dict_to_json(dimension_index_train, os.path.join(folder, DimWiseAdditiveKernelGenerator.dimension_index_file_train))
        write_dict_to_json(dimension_index_val, os.path.join(folder, DimWiseAdditiveKernelGenerator.dimension_index_file_val))
        write_dict_to_json(dimension_index_test, os.path.join(folder, DimWiseAdditiveKernelGenerator.dimension_index_file_test))

    def _generate_simulated_data(self, n_datasets: int, folder: str) -> Tuple[List[str], Dict[int, str]]:
        """
        Generates n_datasets SimulatedDataset objects and stores each object in folder - it returns the
        list of uuids of the generated and stored SimulatedDatasets and a dictionary that maps the
        input dimension to the list of uuid that are associated with Datasets with that dimension (this
        acts as a search index through the list of datasets)
        """
        uuids = []
        input_dimension_index = {}
        for i in range(0, n_datasets):
            simulated_dataset = self.generate_one_sample()
            print(simulated_dataset)
            assert isinstance(simulated_dataset, SimulatedDataset)
            assert simulated_dataset.contains_input_kernel_list()
            uuid = simulated_dataset.id
            input_dimension = simulated_dataset.input_dimension
            if input_dimension in input_dimension_index:
                input_dimension_index[input_dimension].append(uuid)
            else:
                input_dimension_index[input_dimension] = []
            simulated_dataset.save(folder)
            uuids.append(uuid)
        return uuids, input_dimension_index

    def generate_one_sample(self, D: Optional[int] = None):
        if D is None:
            D = self.sample_dimension()
        N = self.sample_dataset_len()
        # sample ground truth kernel
        kernel_list, kernel_expression = self.sample_kernel(D)
        simulated_dataset = self.simulator.create_sample(
            N, self.n_test, kernel_expression, self.observation_noise, self.sample_observation_noise
        )
        simulated_dataset.add_kernel_list_gt(kernel_list)
        if self.add_negative_kernels_train:
            random_uniform = np.random.uniform()
            if random_uniform <= self.fraction_gt_kernels_in_train:
                simulated_dataset.add_input_kernel_list(kernel_list)
            else:
                input_kernel_list, _ = self.sample_kernel(D)
                simulated_dataset.add_input_kernel_list(input_kernel_list)
        else:
            simulated_dataset.add_input_kernel_list(kernel_list)
        return simulated_dataset

    def sample_kernel(self, D: int):
        kernel_list = []
        for d in range(0, D):
            # kernel_len = int(np.random.randint(self.min_max_kernel_len[0], self.min_max_kernel_len[1] + 1, 1))
            kernel_len = self.sample_kernel_len()
            if kernel_len > len(self.base_kerne_type_list):
                kernel_len = len(self.base_kerne_type_list)
            kernel_list_for_dim = convert_list_elements_to_int(
                list(
                    np.random.choice(
                        self.base_kerne_type_list, kernel_len, replace=self.replace_in_kernel_list, p=self.weights_base_kernels
                    )
                )
            )
            kernel_list.append(kernel_list_for_dim)
        kernel_expression = transform_kernel_list_to_expression(kernel_list, add_prior=True)
        return kernel_list, kernel_expression

    def sample_kernel_len(self):
        kernel_len = np.random.geometric(self.kernel_len_geometric_p, 1)
        return kernel_len

    def sample_dataset_len(self):
        N = int(np.random.randint(self.min_max_n[0], self.min_max_n[1] + 1, 1))
        return N

    def sample_dimension(self):
        if self.uniform_dist_for_dimension:
            D = int(np.random.randint(self.min_max_d[0], self.min_max_d[1] + 1, 1))
        else:
            D = int(np.random.geometric(self.num_dimension_geometric_p, 1))
            # if D is bigger than max D sample uniform
            if D > self.min_max_d[1]:
                D = int(np.random.randint(self.min_max_d[0], self.min_max_d[1] + 1, 1))
        return D

    def set_initial_seed(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


if __name__ == "__main__":
    pass
