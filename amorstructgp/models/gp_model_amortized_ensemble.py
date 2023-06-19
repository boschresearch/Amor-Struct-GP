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
from typing import List, Optional, Tuple, Union
from amorstructgp.utils.enums import PredictionQuantity
from amorstructgp.config.nn.amortized_infer_models_configs import (
    BasicDimWiseAdditiveKernelAmortizedModelConfig,
    DimWiseAdditiveKernelWithNoiseAmortizedModelConfig,
)
from amorstructgp.gp.base_kernels import BaseKernelTypes
from amorstructgp.nn.amortized_inference_models import (
    BasicDimWiseAdditiveAmortizedInferenceModel,
    to_input_tensor_of_dim_wise_additive,
)
from amorstructgp.nn.amortized_models_factory import AmortizedModelsFactory
from amorstructgp.models.base_model import BaseModel
import torch
import numpy as np

from amorstructgp.utils.gaussian_mixture_density import EntropyApproximation, GaussianMixtureDensity
from amorstructgp.utils.utils import check_array_bounds


class GPModelAmortizedEnsemble(BaseModel):
    def __init__(
        self,
        prediction_quantity: PredictionQuantity,
        amortized_model_config: BasicDimWiseAdditiveKernelAmortizedModelConfig,
        entropy_approximation: EntropyApproximation,
    ):
        self.prediction_quantity = prediction_quantity
        self.model_config = amortized_model_config
        self.entropy_approximation = entropy_approximation
        self.kernel_list = None
        self.kernel_list_defined_on_all_dimensions = False
        self.cached_x_data = None
        self.cached_y_data = None
        self.infered_kernel_embeddings = None
        self.infered_K_trains = None
        self.infered_noise_variances = None
        self.infered_log_marginal_likelis = None
        self.n_ensemble = 0
        self.fast_batch_inference = True
        self.model: BasicDimWiseAdditiveAmortizedInferenceModel = None
        self.device = torch.device("cpu")

    def load_amortized_model(self, snapshot_file_path: str):
        model_config = self.model_config
        snapshot = torch.load(snapshot_file_path, map_location=self.device)
        if "model" in snapshot:
            model_state_dict = snapshot["model"]
        else:
            model_state_dict = snapshot
        self.model = AmortizedModelsFactory.build(model_config)
        self.model.to(self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

    def set_kernel_list(self, kernel_list: Union[List[List[BaseKernelTypes]], List[List[List[BaseKernelTypes]]]]):
        assert isinstance(kernel_list[0], list)
        if isinstance(kernel_list[0][0], list):
            self.kernel_list_defined_on_all_dimensions = True
        else:
            self.kernel_list_defined_on_all_dimensions = False
        self.kernel_list = kernel_list
        self.n_ensemble = len(self.kernel_list)

    def create_input_kernel_list(
        self, kernel_list: Union[List[List[BaseKernelTypes]], List[List[List[BaseKernelTypes]]]], n_dims: int
    ) -> List[List[List[BaseKernelTypes]]]:
        if self.kernel_list_defined_on_all_dimensions:
            return kernel_list
        else:
            new_kernel_list = []
            for kernel_list_in_ensemble in kernel_list:
                kernel_list_over_dim = [kernel_list_in_ensemble for i in range(0, n_dims)]
                new_kernel_list.append(kernel_list_over_dim)
            return new_kernel_list

    def infer(self, x_data: np.array, y_data: np.array):
        assert self.model is not None
        assert self.kernel_list is not None
        assert check_array_bounds(x_data)
        n_dim = x_data.shape[1]
        kernel_list = self.create_input_kernel_list(self.kernel_list, n_dim)
        print(kernel_list)
        n_data = x_data.shape[0]
        if self.fast_batch_inference:
            X_list = [x_data]
            y_list = [np.squeeze(y_data)]
            model_forward_func = self.model.forward_ensemble
        else:
            X_list = [x_data for i in range(0, self.n_ensemble)]
            y_list = [np.squeeze(y_data) for i in range(0, self.n_ensemble)]
            model_forward_func = self.model.forward
        (
            kernel_mask,
            kernel_embeddings,
            X_unsqueezed,
            y_unsqueezed,
            size_mask,
            dim_mask,
            N,
            size_mask_kernel,
        ) = to_input_tensor_of_dim_wise_additive(kernel_list, self.device, X_list, y_list)
        kernel_embeddings, K_train, nmll, _, noise_variances, _, _, _ = model_forward_func(
            X_unsqueezed,
            y_unsqueezed,
            N,
            size_mask,
            dim_mask,
            kernel_embeddings,
            kernel_mask,
            size_mask_kernel,
            kernel_list,
            device=self.device,
        )
        self.infered_kernel_embeddings = kernel_embeddings  # n_ensemble x D x N_k x d_h
        self.infered_K_trains = K_train  # n_ensemble x N x N
        self.cached_x_data = X_unsqueezed[0].unsqueeze(0)
        self.cached_y_data = y_unsqueezed[0].unsqueeze(0)
        self.infered_noise_variances = noise_variances  # n_ensemble x N
        self.infered_log_marginal_likelis = -1.0 * np.squeeze(nmll.detach().numpy()) * n_data
        max_log_marginal_likeli = np.max(self.infered_log_marginal_likelis)
        # subtract max log marignal likeli for numerical stability of softmax
        unnnormalized_ensemble_weights = np.exp(self.infered_log_marginal_likelis - max_log_marginal_likeli)
        self.ensemble_weights = unnnormalized_ensemble_weights / np.sum(unnnormalized_ensemble_weights)
        print("Negative-Log-marginal-likelihood: {}".format(nmll))

    def predict(self, x_test: np.array) -> Tuple[np.array, np.array]:
        assert x_test.shape[1] == self.cached_x_data.shape[2]
        n_dim = x_test.shape[1]
        kernel_list = self.create_input_kernel_list(self.kernel_list, n_dim)
        pred_mus = []
        pred_f_sigmas = []
        pred_y_sigmas = []
        for i in range(0, self.n_ensemble):
            mu_test, sigma_f_test, sigma_y_test = self.model.predict_head(
                x_test,
                [kernel_list[i]],
                self.infered_kernel_embeddings[i].unsqueeze(0),
                self.cached_x_data,
                self.cached_y_data,
                self.infered_K_trains[i].unsqueeze(0),
                self.infered_noise_variances[i].unsqueeze(0),
                device=self.device,
            )
            pred_mus.append(mu_test)
            pred_f_sigmas.append(sigma_f_test)
            pred_y_sigmas.append(sigma_y_test)
        pred_mus = np.array(pred_mus)
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_sigmas = np.array(pred_f_sigmas)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_sigmas = np.array(pred_y_sigmas)
        assert len(pred_sigmas.shape) == 2
        assert len(pred_mus.shape) == 2
        return pred_mus, pred_sigmas

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,)
        sigma array with shape (n,)
        """
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        print("Ensemble weights:")
        print(self.ensemble_weights)
        n = x_test.shape[0]
        mus_over_inputs = []
        sigmas_over_inputs = []
        for i in range(0, n):
            dist = GaussianMixtureDensity(self.ensemble_weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            mu = dist.mean()
            var = dist.variance()
            mus_over_inputs.append(mu)
            sigmas_over_inputs.append(np.sqrt(var))
        return np.array(mus_over_inputs), np.array(sigmas_over_inputs)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        n = x_test.shape[0]
        log_likelis = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(self.ensemble_weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            log_likeli = gmm_at_test_point.log_likelihood(np.squeeze(y_test[i]))
            log_likelis.append(log_likeli)
        return np.squeeze(np.array(log_likelis))

    def reset_model(self):
        self.cached_x_data = None
        self.cached_y_data = None
        self.infered_kernel_embeddings = None
        self.infered_K_trains = None
        self.infered_noise_variances = None
        self.infered_log_marginal_likelis = None
        self.ensemble_weights = None

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        raise NotImplementedError

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        n = x_test.shape[0]
        entropies = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(self.ensemble_weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            gmm_at_test_point.set_entropy_approx_type(self.entropy_approximation)
            entropy = gmm_at_test_point.entropy()
            entropies.append(entropy)
        return np.array(entropies)
