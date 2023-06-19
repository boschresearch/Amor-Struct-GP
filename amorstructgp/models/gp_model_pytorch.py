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
import copy
from typing import Optional, Tuple

import torch
from amorstructgp.models.base_model import BaseModel
import numpy as np
import gpytorch
from enum import Enum
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from amorstructgp.utils.enums import PredictionQuantity
from amorstructgp.utils.gpytorch_utils import get_gpytorch_exponential_prior
from scipy.stats import norm


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, mean_module, kernel_module):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = mean_module
        self.kernel_module = kernel_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModelPytorch(BaseModel):
    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        initial_likelihood_noise: float,
        fix_likelihood_variance: bool,
        optimize_hps: bool,
        set_prior_on_observation_noise: bool,
        observation_noise_variance_lambda: float,
        add_constant_mean_function: bool,
        prediction_quantity: PredictionQuantity,
        training_iter: int,
        lr: float,
        do_multi_start_optimization: bool,
        n_restarts_multistart: int,
        do_map_estimation: bool,
        do_early_stopping: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.do_multi_start_optimization = do_multi_start_optimization
        if self.do_multi_start_optimization:
            # this done to prevent pyro bug when resampling of kernel hyperparameters
            self.kernel_module = gpytorch.kernels.AdditiveKernel(kernel)
            assert len([prior_name for prior_name in self.kernel_module.named_priors()]) > 0
            assert set_prior_on_observation_noise
        else:
            self.kernel_module = kernel
        if set_prior_on_observation_noise:
            noise_variance_prior = get_gpytorch_exponential_prior(observation_noise_variance_lambda)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_variance_prior)
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = torch.tensor(np.power(initial_likelihood_noise, 2.0))
        if fix_likelihood_variance:
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        else:
            self.likelihood.noise_covar.raw_noise.requires_grad_(True)
        if add_constant_mean_function:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.ZeroMean()
        self.prediction_quantity = prediction_quantity
        self.do_map_estimation = do_map_estimation
        self.n_restarts_multistart = n_restarts_multistart
        self.optimize_hps = optimize_hps
        self.model = None
        self.training_iter = training_iter
        self.do_early_stopping = do_early_stopping
        self.min_iter_early_stopping = 10
        self.lr = lr

    def get_likelihood_noise_variance(self):
        return self.likelihood.noise.detach().numpy()

    def build_model(self, x_data: np.array, y_data: np.array):
        self.model = ExactGPModel(x_data, y_data, self.likelihood, self.mean_module, self.kernel_module)

    def infer(self, x_data: np.array, y_data: np.array):
        x_data = torch.from_numpy(x_data).double()
        y_data = torch.from_numpy(np.squeeze(y_data)).double()
        self.build_model(x_data, y_data)
        if self.optimize_hps:
            if self.do_multi_start_optimization:
                loss_list = []
                state_dict_list = []
                for i in range(0, self.n_restarts_multistart):
                    self.resample_model_parameters()
                    loss = self.ml_type2(x_data, y_data).item()
                    loss_list.append(loss)
                    state_dict_list.append(copy.deepcopy(self.model.state_dict()))
                    print(f"Loss at iteration {i+1}/{self.n_restarts_multistart}: {loss}")
                    print("Parameters of run:")
                    self.print_model_parameters()
                loss_array = np.array(loss_list)
                best_index = np.argmin(loss_array)
                print(f"Best run: {best_index+1}")
                print(f"Loss function best: {loss_array[best_index]}")
                best_state_dict = state_dict_list[best_index]
                self.model.load_state_dict(best_state_dict)
                print("Parameters best:")
                self.print_model_parameters()
            else:
                self.ml_type2(x_data, y_data)

    def eval_log_marginal_likelihood(self, x_data: np.array, y_data: np.array) -> torch.Tensor:
        x_data = torch.from_numpy(x_data).double()
        y_data = torch.from_numpy(np.squeeze(y_data)).double()
        return self.log_marginal_likelihood(x_data, y_data)

    def log_marginal_likelihood(self, x_data: torch.tensor, y_data: torch.tensor):
        function_dist = self.model(x_data)
        output = self.likelihood(function_dist)
        res = output.log_prob(y_data)
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)

    def eval_log_posterior_density(self, x_data: np.array, y_data: np.array) -> torch.Tensor:
        x_data = torch.from_numpy(x_data).double()
        y_data = torch.from_numpy(np.squeeze(y_data)).double()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        output = self.model(x_data)
        mll_value = mll(output, y_data)
        return mll_value

    def resample_model_parameters(self):
        print("-Resample model parameters")
        print("-Previous parameters:")
        self.print_model_parameters()
        kernel = self.model.kernel_module.pyro_sample_from_prior()
        kernel_state_dict = kernel.state_dict()
        self.model.kernel_module.load_state_dict(kernel_state_dict)
        # print([prior for prior in self.model.likelihood.noise_covar._priors])
        success = False
        while not success:
            try:
                self.model.likelihood.noise_covar.sample_from_prior("noise_prior")
                success = True
            except:
                print("Error in noise sampling")
        print("-New parameters:")
        self.print_model_parameters()

    def ml_type2(self, x_data: torch.Tensor, y_data: torch.Tensor):
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for i in range(self.training_iter):
            optimizer.zero_grad()
            loss = self.loss_function(x_data, y_data, mll)
            loss.backward()
            if i % 5 == 0:
                print("Iter %d/%d - Loss: %.3f " % (i + 1, self.training_iter, loss.item()))
            # self.print_model_parameters()
            optimizer.step()
            if i > self.min_iter_early_stopping and self.do_early_stopping:
                current_loss = loss.item()
                if np.allclose(current_loss, previous_loss, rtol=1e-5, atol=1e-5):
                    print(f"Stopping criteria triggered at iteration {i}")
                    break
            previous_loss = loss.item()
        # return final loss
        loss = self.loss_function(x_data, y_data, mll)
        return loss

    def loss_function(self, x_data, y_data, mll):
        if self.do_map_estimation:
            # ExactMarginalLogLikelihood also adds prior in the background if existent
            output = self.model(x_data)
            loss = -mll(output, y_data)
        else:
            loss = -1 * self.log_marginal_likelihood(x_data, y_data)
        return loss

    def predict(self, x_test: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            if self.prediction_quantity == PredictionQuantity.PREDICT_F:
                f_pred = self.model(x_test)
                return f_pred
            elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
                y_pred = self.likelihood(self.model(x_test))
                return y_pred

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        x_test = torch.from_numpy(x_test).double()
        pred_dist = self.predict(x_test)
        mean = pred_dist.mean.numpy()
        covar_matrix = pred_dist.covariance_matrix.detach().numpy()
        std = np.sqrt(np.diag(covar_matrix))

        # mean, std = pred_dist.mean.numpy(), pred_dist.stddev.numpy()
        assert len(mean) == len(x_test) and len(std) == len(x_test)
        return mean, std

    def print_model_parameters(self):
        print("Model parameters:")
        for name, param, constraint in self.model.named_parameters_and_constraints():
            if constraint is not None:
                print(f"Parameter name: {name:55} value = {constraint.transform(param)}")
            else:
                print(f"Parameter name: {name:55} value = {param}")

    def reset_model(self):
        pass

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        return super().estimate_model_evidence(x_data, y_data)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        assert self.prediction_quantity == PredictionQuantity.PREDICT_Y
        pred_mu, pred_std = self.predictive_dist(x_test)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mu), np.squeeze(pred_std))
        return log_likelis

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        return super().entropy_predictive_dist(x_test)
