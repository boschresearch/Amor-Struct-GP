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
from amorstructgp.config.nn.amortized_infer_models_configs import (
    BasicDimWiseAdditiveKernelAmortizedModelConfig,
    DimWiseAdditiveKernelWithNoiseAmortizedModelConfig,
)
from amorstructgp.nn.amortized_inference_models import (
    BasicDimWiseAdditiveAmortizedInferenceModel,
    DimWiseAdditiveKernelWithNoiseAmortizedModel,
    to_input_tensor_of_dim_wise_additive,
)
from amorstructgp.nn.amortized_models_factory import AmortizedModelsFactory
from amorstructgp.utils.utils import check_array_bounds, get_number_of_parameters
from amorstructgp.models.base_model import BaseModel
import numpy as np
from amorstructgp.gp.base_kernels import BaseKernelEvalMode, BaseKernelTypes
from amorstructgp.utils.enums import PredictionQuantity
from scipy.stats import norm
import torch


class GPModelAmortizedStructured(BaseModel):
    def __init__(
        self,
        prediction_quantity: PredictionQuantity,
        amortized_model_config: BasicDimWiseAdditiveKernelAmortizedModelConfig,
        do_warm_start: bool,
        warm_start_steps: int,
        warm_start_lr: float,
        **kwargs
    ):
        self.prediction_quantity = prediction_quantity
        self.model_config = amortized_model_config
        self.do_warm_start = do_warm_start
        self.only_learn_noise_variance_in_warm_start = False
        self.warm_start_steps = warm_start_steps
        self.warm_start_lr = warm_start_lr
        self.kernel_list = None
        self.kernel_list_defined_on_all_dimensions = False
        self.cached_x_data = None
        self.cached_y_data = None
        self.infered_kernel_embeddings = None
        self.infered_untransformed_parameters = None
        self.infered_K_train = None
        self.infered_noise_variance = None
        self.model: BasicDimWiseAdditiveAmortizedInferenceModel = None
        self.device = torch.device("cpu")

    def load_amortized_model(self, snapshot_file_path: str, set_to_eval_mode: bool = True):
        snapshot = torch.load(snapshot_file_path, map_location=self.device)
        if "model" in snapshot:
            model_state_dict = snapshot["model"]
        else:
            model_state_dict = snapshot
        self.build_model(model_state_dict, set_to_eval_mode)

    def build_model(self, model_state_dict, set_to_eval_mode: bool = True):
        model_config = self.model_config
        self.model = AmortizedModelsFactory.build(model_config)
        self.model.set_warm_start_params(self.warm_start_steps, self.warm_start_lr)
        if self.do_warm_start:
            self.model.set_eval_mode(BaseKernelEvalMode.WARM_START)
        self.model.to(self.device)
        self.model.load_state_dict(model_state_dict)
        if set_to_eval_mode:
            self.model.eval()

    def set_do_warm_start(self, do_warm_start: bool):
        self.do_warm_start = do_warm_start
        if do_warm_start:
            self.model.set_eval_mode(BaseKernelEvalMode.WARM_START)
        else:
            self.model.set_eval_mode(BaseKernelEvalMode.STANDARD)

    def set_kernel_list(self, kernel_list: Union[List[BaseKernelTypes], List[List[BaseKernelTypes]]]):
        if isinstance(kernel_list[0], list):
            self.kernel_list_defined_on_all_dimensions = True
        else:
            self.kernel_list_defined_on_all_dimensions = False
        self.kernel_list = kernel_list

    def create_input_kernel_list(
        self, kernel_list: Union[List[BaseKernelTypes], List[List[BaseKernelTypes]]], n_dims: int
    ) -> List[List[List[BaseKernelTypes]]]:
        if self.kernel_list_defined_on_all_dimensions:
            return [kernel_list]
        else:
            kernel_list_over_dim = [kernel_list for i in range(0, n_dims)]
            return [kernel_list_over_dim]

    def get_predicted_parameters(self, x_data: np.array, y_data: np.array):
        assert not self.do_warm_start
        assert self.model is not None
        assert self.kernel_list is not None
        n_dim = x_data.shape[1]
        X_list = [x_data]
        y_list = [np.squeeze(y_data)]
        kernel_list = self.create_input_kernel_list(self.kernel_list, n_dim)
        print(kernel_list)
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
        kernel_parameters, noise_variances = self.model.get_predicted_parameters(
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
        return kernel_parameters, noise_variances

    def infer(self, x_data: np.array, y_data: np.array):
        assert self.model is not None
        assert self.kernel_list is not None
        assert check_array_bounds(x_data)
        n_dim = x_data.shape[1]
        X_list = [x_data]
        y_list = [np.squeeze(y_data)]
        kernel_list = self.create_input_kernel_list(self.kernel_list, n_dim)
        print(kernel_list)
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
        if self.do_warm_start:
            untransformed_kernel_params, K_train, nmll, noise_variances = self.model.warm_start(
                X_unsqueezed,
                y_unsqueezed,
                N,
                size_mask,
                dim_mask,
                kernel_embeddings,
                kernel_mask,
                size_mask_kernel,
                kernel_list,
                self.warm_start_steps,
                self.warm_start_lr,
                device=self.device,
                only_learn_noise_variance=self.only_learn_noise_variance_in_warm_start,
            )
            self.infered_untransformed_parameters = untransformed_kernel_params
            print("Untransformed params:" + str(self.infered_untransformed_parameters))
        else:
            kernel_embeddings, K_train, nmll, _, noise_variances, _, _, _ = self.model.forward(
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
        self.infered_kernel_embeddings = kernel_embeddings
        self.infered_K_train = K_train
        self.cached_x_data = X_unsqueezed
        self.cached_y_data = y_unsqueezed
        self.infered_noise_variance = noise_variances
        print("Negative-Log-marginal-likelihood: {}".format(nmll))

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        mu_test, sigma_f_test, sigma_y_test = self.predict(x_test)
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            return mu_test, sigma_f_test
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            return mu_test, sigma_y_test

    def predict(self, x_test):
        assert x_test.shape[1] == self.cached_x_data.shape[2]
        n_dim = x_test.shape[1]
        kernel_list = self.create_input_kernel_list(self.kernel_list, n_dim)
        mu_test, sigma_f_test, sigma_y_test = self.model.predict_head(
            x_test,
            kernel_list,
            self.infered_kernel_embeddings,
            self.cached_x_data,
            self.cached_y_data,
            self.infered_K_train,
            self.infered_noise_variance,
            self.infered_untransformed_parameters,
            device=self.device,
        )

        return mu_test, sigma_f_test, sigma_y_test

    def get_number_of_parameters(self):
        return get_number_of_parameters(self.model)

    def set_noise_manually(self, noise_variance):
        self.infered_noise_variance = torch.from_numpy(np.array([[noise_variance]]))

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        mu_test, _, sigma_y_test = self.predict(x_test)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(mu_test), np.squeeze(sigma_y_test))
        return log_likelis

    def reset_model(self):
        raise NotImplementedError

    def clear_cache(self):
        self.kernel_list = None
        self.kernel_list_defined_on_all_dimensions = False
        self.cached_x_data = None
        self.cached_y_data = None
        self.infered_kernel_embeddings = None
        self.infered_untransformed_parameters = None
        self.infered_K_train = None
        self.infered_noise_variance = None

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        raise NotImplementedError

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        raise NotImplementedError
