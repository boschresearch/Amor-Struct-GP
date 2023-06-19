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
import json
from typing import List, Tuple
import dill as pickle
from amorstructgp.config.models.gp_model_gpytorch_config import BasicGPModelPytorchConfig, GPModelPytorchMAPConfig
from amorstructgp.config.prior_parameters import PRIOR_SETTING, PriorSettings
from amorstructgp.gp.kernel_grammar import (
    BaseKernelGrammarExpression,
    BaseKernelsLibrary,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
import numpy as np

from amorstructgp.gp.gpytorch_kernels.elementary_kernels_pytorch import LinearKernelPytorch, PeriodicKernelPytorch, RBFKernelPytorch
from amorstructgp.config.kernels.gpytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicLinearKernelPytorchConfig,
    BasicRBFPytorchConfig,
    LinearWithPriorPytorchConfig,
    PeriodicWithPriorPytorchConfig,
    RBFWithPriorPytorchConfig,
)
from amorstructgp.gp.base_kernels import (
    BaseKernelParameterFormat,
    BaseKernelTypes,
    KernelParameterNestedList,
    get_expressions_from_kernel_list,
    get_parameter_value_lists,
)
from amorstructgp.models.gp_model_pytorch import GPModelPytorch
from amorstructgp.models.model_factory import ModelFactory
from amorstructgp.utils.plotter import Plotter
from amorstructgp.utils.plotter2D import Plotter2D
import torch
import os
import gpytorch
import uuid
from amorstructgp.utils.gpytorch_utils import (
    get_gpytorch_kernel_from_expression_and_state_dict,
    get_hp_sample_from_prior_gpytorch_as_state_dict,
    print_gpytorch_parameters,
)
from amorstructgp.gp.base_kernels import transform_kernel_list_to_expression
import matplotlib.pyplot as plt
from amorstructgp.utils.gpytorch_utils import get_gpytorch_exponential_prior
from amorstructgp.config.prior_parameters import NOISE_VARIANCE_EXPONENTIAL_LAMBDA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_simulated_dataset_from_files(folder, uuid):
    file_name_x_data = "x_data_{}.txt".format(uuid)
    file_name_y_data = "y_data_{}.txt".format(uuid)
    file_name_f_data = "f_data_{}.txt".format(uuid)
    file_name_x_test = "x_test_{}.txt".format(uuid)
    file_name_y_test = "y_test_{}.txt".format(uuid)
    file_name_f_test = "f_test_{}.txt".format(uuid)
    file_name_meta_data = "meta_data_{}.json".format(uuid)
    file_name_kernel_state_dict = "kernel_state_dict_{}.pth".format(uuid)
    file_name_kernel_expression = "kernel_expression_{}.p".format(uuid)
    x_data = np.loadtxt(os.path.join(folder, file_name_x_data))
    x_test = np.loadtxt(os.path.join(folder, file_name_x_test))
    if len(x_data.shape) == 1:
        assert len(x_test.shape) == 1
        x_data = np.expand_dims(x_data, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
    y_data = np.expand_dims(np.loadtxt(os.path.join(folder, file_name_y_data)), axis=1)
    y_test = np.expand_dims(np.loadtxt(os.path.join(folder, file_name_y_test)), axis=1)
    f_data = np.expand_dims(np.loadtxt(os.path.join(folder, file_name_f_data)), axis=1)
    f_test = np.expand_dims(np.loadtxt(os.path.join(folder, file_name_f_test)), axis=1)
    with open(os.path.join(folder, file_name_meta_data), "r") as fp:
        meta_data = json.load(fp)
    with open(os.path.join(folder, file_name_kernel_expression), "rb") as fp:
        kernel_expression = pickle.load(fp)
    kernel_state_dict = torch.load(os.path.join(folder, file_name_kernel_state_dict))
    if "prior_setting" in meta_data:
        prior_setting = meta_data["prior_setting"]
    else:
        prior_setting = None
    simulated_data = SimulatedDataset(
        x_data,
        y_data,
        f_data,
        x_test,
        y_test,
        f_test,
        meta_data["observation_noise"],
        kernel_expression,
        meta_data["mll_gt"],
        meta_data["log_postior_density_gt"],
        kernel_state_dict,
        prior_setting,
        meta_data["noise_variance_lambda"],
    )
    if "kernel_list_gt" in meta_data:
        simulated_data.add_kernel_list_gt(meta_data["kernel_list_gt"])
    if "input_kernel_list" in meta_data:
        simulated_data.add_input_kernel_list(meta_data["input_kernel_list"])
    return simulated_data


class SimulatedDataset:
    """
    Data class for one instance of a simulated dataset

    Attributes:
        X: np.array - dataset input with shape [n,d] where n is the number of datapoints and d the input dimension
        y: np.array - dataset output shape [n,1] - with noise
        f: np.array - dataset output shape [n,1] - noiseless
        observation_noise: float - observation noise std dev of added noise
        kernel_expression_gt: BaseKernelGrammarExpression - kernel expression corresponding to the kernel that generated the data
        mll_gt: Marginal likelihood value of the data give the kernel that generated the data
        log_posterior_density_gt: MLL + log prior of the kernel parameters
        kernel_state_dict_gt: torch state_dict containing the parameters of the kernel with which the data was generated
        prior_setting: PriodSettings enum value - stores with which prior setting the data was simulated
        noise_variance_lambda: prior parameter for the observation noise prior
    """

    def __init__(
        self,
        X_data: np.array,
        y_data: np.array,
        f_data: np.array,
        X_test: np.array,
        y_test: np.array,
        f_test: np.array,
        observation_noise: float,
        kernel_expression_gt: BaseKernelGrammarExpression,
        mll_gt: float,
        log_posterior_density_gt: float,
        kernel_state_dict_gt: dict,
        prior_setting: PriorSettings,
        noise_variance_lambda: float,
    ):
        self.id = uuid.uuid4().hex
        self.X_data = X_data
        self.y_data = y_data
        self.f_data = f_data
        self.X_test = X_test
        self.y_test = y_test
        self.f_test = f_test
        self.mu_y_data = np.mean(self.y_data)
        self.sigma_y_data = np.std(self.y_data)
        self.y_data_normalized = (self.y_data - self.mu_y_data) / self.sigma_y_data
        self.y_test_normalized = (self.y_test - self.mu_y_data) / self.sigma_y_data
        assert len(self.X_data.shape) == 2
        assert len(self.y_data.shape) == 2
        assert len(self.X_test.shape) == 2
        assert len(self.y_test.shape) == 2
        assert self.X_data.shape[1] == self.X_test.shape[1]
        self.n = X_data.shape[0]
        self.n_test = X_test.shape[0]
        self.input_dimension = X_data.shape[1]
        self.observation_noise = observation_noise  # likelihood noise std
        self.kernel_expression_gt = kernel_expression_gt
        self.mll_gt = mll_gt
        self.log_posterior_density_gt = log_posterior_density_gt
        self.kernel_state_dict_gt = kernel_state_dict_gt
        self.kernel_list_gt = None
        self.input_kernel_list = None
        self.prior_setting = prior_setting
        self.noise_variance_lambda = noise_variance_lambda

    def add_kernel_list_gt(self, kernel_list_gt: List[List[BaseKernelTypes]]):
        self.kernel_list_gt = kernel_list_gt

    def add_input_kernel_list(self, input_kernel_list: List[List[BaseKernelTypes]]):
        self.input_kernel_list = input_kernel_list

    def get_kernel_list_gt(self) -> List[List[BaseKernelTypes]]:
        return self.kernel_list_gt

    def get_input_kernel_list(self) -> List[List[BaseKernelTypes]]:
        if self.contains_input_kernel_list():
            return self.input_kernel_list
        else:
            return self.kernel_list_gt

    def contains_input_kernel_list(self) -> bool:
        if self.input_kernel_list is None:
            return False
        return True

    def get_num_datapoints(self) -> int:
        return self.n

    def get_input_dimension(self) -> int:
        return self.input_dimension

    def get_observation_noise(self) -> float:
        return self.observation_noise

    def get_kernel_expression_gt(self) -> BaseKernelGrammarExpression:
        return self.kernel_expression_gt

    def get_kernel_state_dict_gt(self):
        return self.kernel_state_dict_gt

    def get_dataset(self, normalized_version=False) -> Tuple[np.array, np.array]:
        if normalized_version:
            return self.X_data, self.y_data_normalized
        return self.X_data, self.y_data

    def get_ground_truth_f(self, normalized_version=False) -> Tuple[np.array, np.array]:
        if normalized_version:
            f_data_normalized = (self.f_data - self.mu_y_data) / self.sigma_y_data
            f_test_normalized = (self.f_test - self.mu_y_data) / self.sigma_y_data
            return f_data_normalized, f_test_normalized
        return self.f_data, self.f_test

    def get_gt_kernel_parameter_list(self) -> KernelParameterNestedList:
        value_list = get_parameter_value_lists(self.kernel_list_gt, self.kernel_state_dict_gt)
        return value_list

    def get_test_dataset(self, normalized_version=False) -> Tuple[np.array, np.array]:
        if normalized_version:
            return self.X_test, self.y_test_normalized
        return self.X_test, self.y_test

    def save(self, folder):
        file_name_x_data = "x_data_{}.txt".format(self.id)
        file_name_y_data = "y_data_{}.txt".format(self.id)
        file_name_f_data = "f_data_{}.txt".format(self.id)
        file_name_x_test = "x_test_{}.txt".format(self.id)
        file_name_y_test = "y_test_{}.txt".format(self.id)
        file_name_f_test = "f_test_{}.txt".format(self.id)
        file_name_meta_data = "meta_data_{}.json".format(self.id)
        file_name_kernel_state_dict = "kernel_state_dict_{}.pth".format(self.id)
        file_name_kernel_expression = "kernel_expression_{}.p".format(self.id)
        np.savetxt(os.path.join(folder, file_name_x_data), self.X_data)
        np.savetxt(os.path.join(folder, file_name_y_data), self.y_data)
        np.savetxt(os.path.join(folder, file_name_f_data), self.f_data)
        np.savetxt(os.path.join(folder, file_name_x_test), self.X_test)
        np.savetxt(os.path.join(folder, file_name_y_test), self.y_test)
        np.savetxt(os.path.join(folder, file_name_f_test), self.f_test)
        meta_data_dict = self.get_meta_data_dict()
        with open(os.path.join(folder, file_name_meta_data), "w") as fp:
            json.dump(meta_data_dict, fp)
        torch.save(self.kernel_state_dict_gt, os.path.join(folder, file_name_kernel_state_dict))
        with open(os.path.join(folder, file_name_kernel_expression), "wb") as fp:
            pickle.dump(self.kernel_expression_gt, fp)

    def get_meta_data_dict(self):
        meta_data = {}
        meta_data["observation_noise"] = self.observation_noise
        meta_data["mll_gt"] = self.mll_gt
        meta_data["log_postior_density_gt"] = self.log_posterior_density_gt
        meta_data["n"] = self.n
        meta_data["n_test"] = self.n_test
        meta_data["input_dimension"] = int(self.input_dimension)
        meta_data["prior_setting"] = self.prior_setting
        meta_data["noise_variance_lambda"] = self.noise_variance_lambda
        if self.kernel_list_gt is not None:
            meta_data["kernel_list_gt"] = self.kernel_list_gt
        if self.input_kernel_list is not None:
            meta_data["input_kernel_list"] = self.input_kernel_list

        return meta_data

    def __str__(self):
        kernel_describtion = self.kernel_expression_gt.get_name()
        describtion = "simulated-dataset - n={} - dim={} - noise-std={} - mll={} - log-density={} - kernel_expression={}".format(
            self.n, self.input_dimension, self.observation_noise, self.mll_gt, self.log_posterior_density_gt, kernel_describtion
        )
        return describtion

    def plot(self):
        if self.input_dimension == 1:
            plotter = Plotter(1)
            plotter.add_datapoints(np.squeeze(self.X_data), np.squeeze(self.y_data), "green", 0)
            plotter.add_datapoints(np.squeeze(self.X_test), np.squeeze(self.y_test), "red", 0)
            plotter.show()
        elif self.input_dimension == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")

            # ax.plot_trisurf(np.squeeze(xs[:, 0]), np.squeeze(xs[:, 1]), np.squeeze(ys), linewidth=1.2, cmap="viridis")
            ax.scatter(self.X_data[:, 0], self.X_data[:, 1], np.squeeze(self.y_data), marker=".", color="green")
            ax.scatter(self.X_test[:, 0], self.X_test[:, 1], np.squeeze(self.y_test), marker=".", color="red")
            plt.show()
        else:
            raise NotImplementedError


class Simulator:
    """
    Class to draw data from a GP prior
    """

    def __init__(self, box_bound_lower: float, box_bound_upper: float, observation_noise_variance_lambda: float):
        self.box_bound_lower = box_bound_lower
        self.box_bound_upper = box_bound_upper
        self.observation_noise_variance_lambda = observation_noise_variance_lambda
        self.noise_variance_pior = get_gpytorch_exponential_prior(observation_noise_variance_lambda)
        self.noise_variance_lower_bound = 1e-4

    def create_sample(
        self,
        n_data: int,
        n_test: int,
        kernel_expression: BaseKernelGrammarExpression,
        observation_noise: float = 0.01,
        sample_observation_noise: float = False,
        make_deep_copy_of_expression: bool = True,
    ) -> SimulatedDataset:
        """
        Main method to create a sample dataset from a GP -  it gets a kernel expression with pytorch base kernels with priors -
        draws a prior sample from the kernel parameters - draws dataset locations X uniformly - draws f from the GP prior
        given the kernel parameters - adds Gaussian noise to f with std_dev=observation_noise to create y
        The method also calculates the marginal likelihood under the GP and the log-posterior density under the GP with kernel parameter priors
        it stores everything in a SimulatedDataset object.
        """
        if make_deep_copy_of_expression:
            kernel_expression = kernel_expression.deep_copy()
        assert kernel_expression.get_base_kernel_library() == BaseKernelsLibrary.GPYTORCH
        input_dimension = kernel_expression.get_input_dimension()
        n = n_data + n_test

        # @TODO: Better solution for kernel sampling
        sampling_success = False  # numerical errors can happen in multivariate normal -> resample if that happens
        while not sampling_success:
            try:
                X = np.random.uniform(self.box_bound_lower, self.box_bound_upper, size=(n, input_dimension))
                X_torch = torch.from_numpy(X)
                kernel_state_dict = get_hp_sample_from_prior_gpytorch_as_state_dict(kernel_expression, wrap_in_addition=True)
                kernel = get_gpytorch_kernel_from_expression_and_state_dict(kernel_expression, kernel_state_dict, wrap_in_addition=True)
                K = kernel(X_torch).numpy()
                f = np.squeeze(np.random.multivariate_normal(np.zeros(n), K, 1))
                if sample_observation_noise:
                    observation_noise_variance = self.noise_variance_pior.sample().numpy()
                    # add lower bound to ensure variance is greater
                    observation_noise_variance = observation_noise_variance + self.noise_variance_lower_bound
                    observation_noise = np.sqrt(observation_noise_variance)

                noise = np.random.normal(0.0, observation_noise, n)
                y = f + noise
                # f = np.expand_dims(f, axis=1)
                y = np.expand_dims(y, axis=1)
                f = np.expand_dims(f, axis=1)
                X_data = X[:n_data]
                y_data = y[:n_data]
                f_data = f[:n_data]
                assert len(X_data) == n_data and len(y_data) == n_data
                X_test = X[n_data:]
                y_test = y[n_data:]
                f_test = f[n_data:]
                assert len(X_test) == n_test and len(y_test) == n_test
                model_config = BasicGPModelPytorchConfig(kernel_config=BasicRBFPytorchConfig(input_dimension=0))
                model_config.optimize_hps = False
                model_config.set_prior_on_observation_noise = sample_observation_noise
                model_config.observation_noise_variance_lambda = self.observation_noise_variance_lambda
                model_config.initial_likelihood_noise = observation_noise
                model_config.add_constant_mean_function = False
                gp_model = GPModelPytorch(kernel=kernel, **model_config.dict())
                y_data_torch = np.squeeze(y_data)
                X_data_torch = X_data
                gp_model.infer(X_data_torch, y_data_torch)
                mll = gp_model.eval_log_marginal_likelihood(X_data_torch, y_data_torch)
                log_posterior_density = gp_model.eval_log_posterior_density(X_data_torch, y_data_torch)
                sampling_success = True
            except:
                print("Error in sampling happened")
        simulated_dataset = SimulatedDataset(
            X_data,
            y_data,
            f_data,
            X_test,
            y_test,
            f_test,
            float(observation_noise),
            kernel_expression.deep_copy(),
            float(mll),
            float(log_posterior_density),
            kernel_state_dict,
            PRIOR_SETTING,
            self.observation_noise_variance_lambda,
        )
        return simulated_dataset


if __name__ == "__main__":
    pass
