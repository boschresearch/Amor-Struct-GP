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
from datetime import datetime
import json
import random
from typing import List, Union
import gpytorch
import numpy as np
import torch
from amorstructgp.gp.kernel_grammar import BaseKernelGrammarExpression
from amorstructgp.nn.dataset_encoder import get_padded_dataset_and_masks
from amorstructgp.gp.base_kernels import get_kernel_embeddings_and_kernel_mask, BaseKernelTypes
from amorstructgp.utils.plotter import Plotter
from amorstructgp.utils.plotter2D import Plotter2D
from scipy.stats import norm


def get_test_inputs(batch_size, max_n, max_d, max_len_kernel, only_SE=False):
    X_list = []
    y_list = []
    if only_SE:
        kernel_type_list = [BaseKernelTypes.SE]
    else:
        kernel_type_list = [
            BaseKernelTypes.SE,
            BaseKernelTypes.LIN,
            BaseKernelTypes.PER,
            BaseKernelTypes.LIN_MULT_PER,
            BaseKernelTypes.SE_MULT_LIN,
            BaseKernelTypes.SE_MULT_PER,
            BaseKernelTypes.MATERN52,
            BaseKernelTypes.SE_MULT_MATERN52,
            BaseKernelTypes.LIN_MULT_MATERN52,
            BaseKernelTypes.PER_MULT_MATERN52,
        ]
    kernel_list_over_batch = []
    for b in range(batch_size):
        N = np.random.randint(2, max_n)
        dim = np.random.randint(1, max_d)
        X = np.random.rand(N, dim)
        y = np.random.rand(N)
        X_list.append(X)
        y_list.append(y)
        kernel_list_dataset = []
        for d in range(dim):
            if max_len_kernel == 1:
                n_kernels = 1
            else:
                n_kernels = np.random.randint(1, max_len_kernel)
            kernel_list_dim = convert_list_elements_to_int(list(np.random.choice(kernel_type_list, n_kernels)))
            kernel_list_dataset.append(kernel_list_dim)
        kernel_list_over_batch.append(kernel_list_dataset)
    print(kernel_list_over_batch[0][0][0])
    kernel_embeddings, kernel_mask = get_kernel_embeddings_and_kernel_mask(kernel_list_over_batch)
    X_padded, y_padded, size_mask, dim_mask, N, size_mask_kernel = get_padded_dataset_and_masks(X_list, y_list)
    kernel_embeddings = torch.from_numpy(kernel_embeddings).float()
    kernel_mask = torch.from_numpy(kernel_mask).float()
    X_padded = torch.from_numpy(X_padded).float()
    y_padded = torch.from_numpy(y_padded).float()
    size_mask = torch.from_numpy(size_mask).float()
    dim_mask = torch.from_numpy(dim_mask).float()
    size_mask_kernel = torch.from_numpy(size_mask_kernel).float()
    N = torch.from_numpy(N).float()
    return (
        X_list,
        y_list,
        X_padded,
        y_padded,
        N,
        size_mask,
        dim_mask,
        kernel_embeddings,
        kernel_mask,
        size_mask_kernel,
        kernel_list_over_batch,
    )


def cut_string(string, interval):
    length = len(string)
    n_splits = int(length / interval)
    new_string = ""
    for i in range(0, n_splits):
        new_string = new_string + string[:interval] + "\n "
        string = string[interval:]
    new_string += string
    return new_string


def plot_predictions(
    X_data,
    y_data1,
    y_data2,
    y_data3,
    X_test,
    y_test1,
    y_test2,
    y_test3,
    X_eval,
    pred_mu_1,
    pred_sigma_1,
    pred_mu_2,
    pred_sigma_2,
    pred_mu_3,
    pred_sigma_3,
    title1,
    title2,
    title3,
    kernel_describtion,
):
    assert len(X_data.shape) == 2
    input_dimension = X_data.shape[1]
    if input_dimension == 1:
        plotter = Plotter(3)
        plotter.add_datapoints(np.squeeze(X_data), np.squeeze(y_data1), "green", 0)
        plotter.add_datapoints(np.squeeze(X_test), np.squeeze(y_test1), "red", 0)
        plotter.add_predictive_dist(np.squeeze(X_eval), np.squeeze(pred_mu_1), np.squeeze(pred_sigma_1), 0)
        plotter.give_axes(0).set_title(title1)
        plotter.add_datapoints(np.squeeze(X_data), np.squeeze(y_data2), "green", 1)
        plotter.add_datapoints(np.squeeze(X_test), np.squeeze(y_test2), "red", 1)
        plotter.add_predictive_dist(np.squeeze(X_eval), np.squeeze(pred_mu_2), np.squeeze(pred_sigma_2), 1)
        plotter.give_axes(1).set_title(title2)
        plotter.add_datapoints(np.squeeze(X_data), np.squeeze(y_data3), "green", 2)
        plotter.add_datapoints(np.squeeze(X_test), np.squeeze(y_test3), "red", 2)
        plotter.add_predictive_dist(np.squeeze(X_eval), np.squeeze(pred_mu_3), np.squeeze(pred_sigma_3), 2)
        plotter.give_axes(2).set_title(title3)
        plotter.fig.suptitle(cut_string(kernel_describtion, 100))
        plotter.fig.set_size_inches(15.0, 15.0)
        plotter.fig.tight_layout()
        return plotter.fig
    elif input_dimension == 2:
        plotter = Plotter2D(3)
        plotter.add_datapoints(np.squeeze(X_data), "green", 0)
        plotter.add_datapoints(np.squeeze(X_test), "red", 0)
        plotter.add_gt_function(np.squeeze(X_eval), np.squeeze(pred_mu_1), "seismic", 100, 0)
        plotter.give_axes(0).set_title(title1)
        plotter.add_datapoints(np.squeeze(X_data), "green", 1)
        plotter.add_datapoints(np.squeeze(X_test), "red", 1)
        plotter.add_gt_function(np.squeeze(X_eval), np.squeeze(pred_mu_2), "seismic", 100, 1)
        plotter.give_axes(1).set_title(title2)
        plotter.add_datapoints(np.squeeze(X_data), "green", 2)
        plotter.add_datapoints(np.squeeze(X_test), "red", 2)
        plotter.add_gt_function(np.squeeze(X_eval), np.squeeze(pred_mu_3), "seismic", 100, 2)
        plotter.give_axes(2).set_title(title3)
        plotter.fig.suptitle(cut_string(kernel_describtion, 100))
        plotter.fig.set_size_inches(20.0, 10.0)
        plotter.fig.tight_layout()
        return plotter.fig


def convert_list_elements_to_int(original_list):
    return [int(element) for element in original_list]


def get_number_of_parameters(model):
    n_params = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            n_params += parameter.numel()
    return n_params


def default_worker_init_function(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_list_from_file(file_path: str):
    with open(file_path, "r") as f:
        alist = f.read().splitlines()
    return alist


def read_dict_from_json(file_path: str):
    with open(file_path, "r") as fp:
        dictionary = json.load(fp)
    return dictionary


def write_list_to_file(alist: List[Union[float, int, str]], file_path: str):
    with open(file_path, "w") as f:
        for line in alist:
            f.write(f"{line}\n")


def write_dict_to_json(dictionary, file_path: str):
    with open(file_path, "w") as f:
        json.dump(dictionary, f)


def get_datetime_as_string():
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    return date_time


def calculate_rmse(y_pred, y_true):
    return np.sqrt(np.mean(np.power(np.squeeze(y_pred) - np.squeeze(y_true), 2.0)))


def calculate_nll_normal(y_test, pred_mus, pred_sigmas):
    return -1 * norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mus), np.squeeze(pred_sigmas))


def check_array_bounds(arr):
    return np.all((arr >= 0.0) & (arr <= 1.0))


if __name__ == "__main__":
    pass
