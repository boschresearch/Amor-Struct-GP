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
from abc import abstractmethod
from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from amorstructgp.gp.rbf_kernel_wrapper import BatchedRBFKernelWrapper
from amorstructgp.nn.dataset_encoder import DatasetEncoder, get_padded_dataset_and_masks
from amorstructgp.nn.kernel_encoder import KernelEncoder
from amorstructgp.gp.base_kernels import (
    BaseKernelEvalMode,
    BaseKernelTypes,
    BatchedDimWiseAdditiveKernelWrapper,
    KernelParameterNestedList,
    KernelTypeList,
    get_kernel_embeddings_and_kernel_mask,
)
from amorstructgp.gp.utils import GP_noise, cal_marg_likelihood, cal_marg_likelihood_batch_noise
from amorstructgp.nn.noise_predictor_head import (
    NoiseVariancePredictorHead,
    BaseNoiseVariancePredictorHead,
)
from amorstructgp.utils.utils import get_test_inputs
from amorstructgp.gp.base_kernels import flatten_nested_list


class BasicDimWiseAdditiveAmortizedInferenceModel(nn.Module):
    def __init__(
        self,
        dataset_encoder: DatasetEncoder,
        kernel_encoder: KernelEncoder,
        share_weights_in_additions: bool,
        kernel_wrapper_hidden_layer_list: List[int],
        kernel_wrapper_dropout_p: float,
        eval_mode=BaseKernelEvalMode.STANDARD,
    ) -> None:
        super().__init__()
        self.eval_mode = eval_mode
        self.dataset_encoder = dataset_encoder
        self.kernel_encoder = kernel_encoder
        self.kernel_wrapper = BatchedDimWiseAdditiveKernelWrapper(
            self.kernel_encoder.hidden_dim,
            kernel_wrapper_hidden_layer_list,
            kernel_wrapper_dropout_p,
            share_weights_in_additions,
            eval_mode=eval_mode,
        )
        self.warm_start_steps = None
        self.warm_start_lr = None

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        self.eval_mode = eval_mode
        self.kernel_wrapper.set_eval_mode(eval_mode)

    def set_warm_start_params(self, num_steps: int, lr: float):
        self.warm_start_steps = num_steps
        self.warm_start_lr = lr

    def freeze_dataset_encoder(self):
        for param in self.dataset_encoder.parameters():
            param.requires_grad = False

    def unfreeze_dataset_encoder(self):
        for param in self.dataset_encoder.parameters():
            param.requires_grad = True

    @abstractmethod
    def get_num_predicted_params(self, kernel_list: List[List[List[BaseKernelTypes]]], device=torch.device("cpu")) -> torch.Tensor:
        """
        Returns the number of gp model parameters (kernel + likelihood) that are predicted by the amortization network
        aka the output dimension for each batch element - fixed parameters are not considered e.g. when the likelihood variance is fixed
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        X_data: B X N X D
        Y_data: B X N
        N_data: B
        size_mask: B X N
        dim_mask: B X D
        kernel_embeddings B X D X N_k X d_k
        kernel_mask_enc B X D x N_k
        size_mask_kernel B x N x N
        kernel_list list of BaseKernelTypes - first list over batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]


        Return:
        kernel_embeddings torch.Tensor # B X D X N_k x hidden_dim kernel enc - learned kernel embeddings
        K torch.Tensor B x N x N - gram matrix of learned kernel
        nmll - torch.Tensor B - vector of negative mean marginal likelihoods
        nmll_with_prior - torch.Tensor B - vector of negative mean log-likelihood of model with priors on kernel parameters
        noise_variance - torch.Tensor Bx1 - vector of predicted noise variance
        log_prior_prob_kernel_params - log prior value of predicted kernel params
        log_prior_prob_variance - log prior value of predicted variance
        mll_success - bool - flag if mll calc was succesfull for all batch elements - is false if chol error occured
        """
        raise NotImplementedError

    @abstractmethod
    def forward_ensemble(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        X_data: 1 X N X D
        Y_data: 1 X N
        N_data: 1
        size_mask: 1 X N
        dim_mask: 1 X D
        kernel_embeddings B_ens X D X N_k X d_k
        kernel_mask B_ens X D x N_k
        kernel_list list of BaseKernelTypes - first list over ensemble batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]


        Return:
        kernel_embeddings torch.Tensor # B_ens X D X N_k x hidden_dim kernel enc - learned kernel embeddings
        K torch.Tensor B_ens x N x N - gram matrix of learned kernel
        nmll - torch.Tensor B_ens - vector of negative mean marginal likelihoods
        nmll_with_prior - torch.Tensor B_ens - vector of negative mean log-likelihood of model with priors on kernel parameters
        noise_variance - torch.Tensor B_ens x 1 - vector of predicted noise variance
        log_prior_prob_kernel_params - log prior value of predicted kernel params
        log_prior_prob_variance - log prior value of predicted variance
        mll_success - bool - flag if mll calc was succesfull for all batch elements - is false if chol error occured
        """
        raise NotImplementedError

    @abstractmethod
    def warm_start(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        num_steps: int,
        learning_rate: float,
        device=torch.device("cpu"),
        only_learn_noise_variance: bool = False,
    ):
        """
        X_data: B X N X D
        Y_data: B X N
        N_data: B
        size_mask: B X N
        dim_mask: B X D
        kernel_embeddings B X D X N_k X d_k
        kernel_mask_enc B X D x N_k
        size_mask_kernel B x N x N
        kernel_list list of BaseKernelTypes - first list over batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
        """
        raise NotImplementedError

    def get_parameter_nested_lists(
        self, kernel_embeddings: torch.Tensor, kernel_list: List[KernelTypeList], detach: bool = True
    ) -> List[KernelParameterNestedList]:
        # kernel_embeddings torch.Tensor # B X D X N_k x hidden_dim kernel enc - learned kernel embeddings
        return self.kernel_wrapper.get_parameters(kernel_embeddings, kernel_list, detach)

    def get_predicted_parameters(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ):
        kernel_embeddings, _, _, _, noise_variances, _, _, _ = self.forward(
            X_data, Y_data, N_data, size_mask, dim_mask, kernel_embeddings, kernel_mask_enc, size_mask_kernel, kernel_list, device
        )
        kernel_parameters = self.get_parameter_nested_lists(kernel_embeddings, kernel_list, detach=False)
        return kernel_parameters, noise_variances

    def predict(
        self, X_test: np.array, X_data: np.array, Y_data: np.array, kernel_list: List[List[BaseKernelTypes]], device=torch.device("cpu")
    ):
        """
        High level API for prediction

        X_test: np.array N_test x D
        X_data: np.array N_train x D
        Y_data: np.array N_train x 1
        """

        assert len(X_test.shape) == 2
        assert len(X_data.shape) == 2
        assert len(Y_data.shape) == 2
        assert isinstance(kernel_list[0][0], BaseKernelTypes) or isinstance(kernel_list[0][0], int)
        kernel_list = [kernel_list]
        X_list = [X_data]
        y_list = [np.squeeze(Y_data)]
        (
            kernel_mask,
            kernel_embeddings,
            X_unsqueezed,
            y_unsqueezed,
            size_mask,
            dim_mask,
            N,
            size_mask_kernel,
        ) = to_input_tensor_of_dim_wise_additive(kernel_list, device, X_list, y_list)
        if self.eval_mode == BaseKernelEvalMode.WARM_START:
            untransformed_kernel_params, K_train, _, noise_variances = self.warm_start(
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
                device=device,
            )
        else:
            kernel_embeddings, K_train, _, _, noise_variances, _, _, _ = self.forward(
                X_unsqueezed,
                y_unsqueezed,
                N,
                size_mask,
                dim_mask,
                kernel_embeddings,
                kernel_mask,
                size_mask_kernel,
                kernel_list,
                device=device,
            )
            untransformed_kernel_params = None
        mu_test, sigma_f_test, sigma_y_test = self.predict_head(
            X_test,
            kernel_list,
            kernel_embeddings,
            X_unsqueezed,
            y_unsqueezed,
            K_train,
            noise_variances,
            untransformed_kernel_params,
            device=device,
        )
        return mu_test, sigma_f_test, sigma_y_test

    def predict_head(
        self,
        X_test: np.array,
        kernel_list: List[List[List[BaseKernelTypes]]],
        kernel_embeddings: torch.Tensor,
        X_unsqueezed: torch.Tensor,
        y_unsqueezed: torch.Tensor,
        K_train: torch.Tensor,
        noise_variances: torch.Tensor,
        untransformed_kernel_params=None,
        device=torch.device("cpu"),
    ):
        """
        Calculates predictive distribution at test points given inferred kernel embeddings and test vector
        Method has been splitted from predict method to enable multiple predictions without recalculating kernel embeddings (reestimating kernel parameters)

        X_test: np.array N_test x D
        kernel_list: List[List[List[BaseKernelTypes]]]
        kernel_embeddings: torch.Tensor (output of self.forward)
        X_unsqueezed: torch.Tensor 1 x N_train x D
        y_unsqueezed: torch.Tensor 1 x N_train
        K_train: torch.Tensor 1 x N_train x N_train (output of self.forward)
        noise_variances: torch.Tensor 1 x 1 (output of self.forward)
        """
        if self.eval_mode == BaseKernelEvalMode.WARM_START:
            assert untransformed_kernel_params is not None
        else:
            assert untransformed_kernel_params is None
        assert len(X_test.shape) == 2
        assert len(X_unsqueezed.shape) == 3
        assert len(y_unsqueezed.shape) == 2
        assert X_unsqueezed.shape[0] == 1
        assert y_unsqueezed.shape[0] == 1
        # assert isinstance(kernel_list[0][0], BaseKernelTypes) or isinstance(kernel_list[0][0], int)
        X_test = torch.from_numpy(X_test).float().to(device)
        X_test_unsqueezed = X_test.unsqueeze(0)  # 1 x N_test x D
        K_train_test = self.kernel_wrapper.forward(
            X_unsqueezed, X_test_unsqueezed, kernel_embeddings, kernel_list, untransformed_kernel_params
        )  # 1 x N_train x N_test
        K_test = self.kernel_wrapper.forward(
            X_test_unsqueezed, X_test_unsqueezed, kernel_embeddings, kernel_list, untransformed_kernel_params
        )  # 1 x N_test x N_test
        noise_variance = noise_variances.squeeze()
        mu_test, covar_test = GP_noise(
            y_unsqueezed.squeeze(0), K_train.squeeze(0), K_train_test.squeeze(0), K_test.squeeze(0), noise_variance, device
        )
        mu_test = mu_test.detach().cpu().numpy()  # shape N_test
        covar_test = covar_test.detach().cpu().numpy()
        noise_variance = noise_variance.detach().cpu().numpy()
        var_f_test = np.diag(covar_test)
        var_y_test = var_f_test + noise_variance
        sigma_f_test = np.sqrt(var_f_test)  # shape N_test
        sigma_y_test = np.sqrt(var_y_test)  # shape N_test
        return mu_test, sigma_f_test, sigma_y_test


class DimWiseAdditiveKernelsAmortizedInferenceModel(BasicDimWiseAdditiveAmortizedInferenceModel):
    def __init__(
        self,
        dataset_encoder: DatasetEncoder,
        kernel_encoder: KernelEncoder,
        share_weights_in_additions: bool,
        kernel_wrapper_hidden_layer_list: List[int],
        kernel_wrapper_dropout_p: float,
        epsilon: float,
        eval_mode=BaseKernelEvalMode.STANDARD,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_encoder,
            kernel_encoder,
            share_weights_in_additions,
            kernel_wrapper_hidden_layer_list,
            kernel_wrapper_dropout_p,
            eval_mode,
        )
        self.epsilon = epsilon  # gp variance

    def get_num_predicted_params(self, kernel_list: List[List[List[BaseKernelTypes]]], device=torch.device("cpu")):
        n_kernel_params = self.kernel_wrapper.get_num_params(kernel_list)
        n_kernel_params = torch.Tensor(n_kernel_params).to(device)
        return n_kernel_params

    def forward(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ):
        """
        X_data: B X N X D
        Y_data: B X N
        N_data: B
        size_mask: B X N
        dim_mask: B X D
        kernel_embeddings B X D X N_k X d_k
        kernel_mask_enc B X D x N_k
        size_mask_kernel B x N x N
        kernel_list list of BaseKernelTypes - first list over batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
        """
        batch_size = X_data.shape[0]
        dataset_embedding = self.dataset_encoder.forward(X_data, Y_data, size_mask, dim_mask, device)  # B X D X hidden_dim_1
        kernel_embeddings = self.kernel_encoder.forward(
            kernel_embeddings, dataset_embedding, kernel_mask_enc, dim_mask
        )  # B X D X N_k x hidden_dim_2
        K = self.kernel_wrapper.forward(X_data, X_data, kernel_embeddings, kernel_list)  # B x N x N
        log_prior_prob_kernel_params = self.kernel_wrapper.get_log_prior_prob(kernel_embeddings, kernel_list)  # B
        mll, mll_success = cal_marg_likelihood(K, Y_data.unsqueeze(-1), self.epsilon, size_mask_kernel, 1.0 - size_mask, N_data, device)
        if mll_success:
            nmll = -1.0 * mll
            nmll_with_prior = nmll - log_prior_prob_kernel_params / N_data
        else:
            nmll = None
            nmll_with_prior = None
        # return fix noise variances as tensor over batch
        noise_variances = torch.tensor(self.epsilon).to(device).repeat(batch_size).unsqueeze(-1)  # Bx1
        return kernel_embeddings, K, nmll, nmll_with_prior, noise_variances, log_prior_prob_kernel_params, None, mll_success

    def forward_ensemble(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        X_data: 1 X N X D
        Y_data: 1 X N
        N_data: 1
        size_mask: 1 X N
        dim_mask: 1 X D
        kernel_embeddings B_ens X D X N_k X d_k
        kernel_mask B_ens X D x N_k
        kernel_list list of BaseKernelTypes - first list over ensemble batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
        """
        raise NotImplementedError


class DimWiseAdditiveKernelWithNoiseAmortizedModel(BasicDimWiseAdditiveAmortizedInferenceModel):
    def __init__(
        self,
        dataset_encoder: DatasetEncoder,
        kernel_encoder: KernelEncoder,
        noise_variance_predictor: BaseNoiseVariancePredictorHead,
        share_weights_in_additions: bool,
        kernel_wrapper_hidden_layer_list: List[int],
        kernel_wrapper_dropout_p: float,
        eval_mode=BaseKernelEvalMode.STANDARD,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_encoder,
            kernel_encoder,
            share_weights_in_additions,
            kernel_wrapper_hidden_layer_list,
            kernel_wrapper_dropout_p,
            eval_mode,
        )
        self.noise_variance_predictor = noise_variance_predictor
        self.noise_variance_predictor.set_eval_mode(self.eval_mode)

    def get_num_predicted_params(self, kernel_list: List[List[List[BaseKernelTypes]]], device=torch.device("cpu")):
        n_kernel_params = self.kernel_wrapper.get_num_params(kernel_list)
        n_kernel_params = torch.Tensor(n_kernel_params).to(device)  # B
        n_gp_params = n_kernel_params + 1.0
        return n_gp_params

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        super().set_eval_mode(eval_mode)
        self.noise_variance_predictor.set_eval_mode(self.eval_mode)

    def forward(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ):
        """
        X_data: B X N X D
        Y_data: B X N
        N_data: B
        size_mask: B X N
        dim_mask: B X D
        kernel_embeddings B X D X N_k X d_k
        kernel_mask_enc B X D x N_k
        size_mask_kernel B x N x N
        kernel_list list of BaseKernelTypes - first list over batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
        """
        dataset_embedding = self.dataset_encoder.forward(X_data, Y_data, size_mask, dim_mask, device)  # B X D X hidden_dim_1
        kernel_embeddings = self.kernel_encoder.forward(
            kernel_embeddings, dataset_embedding, kernel_mask_enc, dim_mask
        )  # B X D X N_k x hidden_dim_2
        K = self.kernel_wrapper.forward(X_data, X_data, kernel_embeddings, kernel_list)  # B x N x N
        log_prior_prob_kernel_params = self.kernel_wrapper.get_log_prior_prob(kernel_embeddings, kernel_list)  # B
        noise_variances, _, log_prior_prob_variance = self.noise_variance_predictor.forward(
            kernel_embeddings, kernel_mask_enc, dataset_embedding, dim_mask
        )  # B x 1, _, B
        mll, mll_success = cal_marg_likelihood_batch_noise(
            K, Y_data.unsqueeze(-1), noise_variances, size_mask_kernel, 1.0 - size_mask, N_data, device
        )  # B
        # chol in mll might fail due to numerical instabilites - some upstream methods might not use the mll such as predict --> we dont throw an error here but push the information upstream
        if mll_success:
            nmll = -1.0 * mll
            nmll_with_prior = nmll - (log_prior_prob_kernel_params + log_prior_prob_variance) / N_data
        else:
            nmll = None
            nmll_with_prior = None

        return (
            kernel_embeddings,
            K,
            nmll,
            nmll_with_prior,
            noise_variances,
            log_prior_prob_kernel_params,
            log_prior_prob_variance,
            mll_success,
        )

    def forward_ensemble(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        X_data: 1 X N X D
        Y_data: 1 X N
        N_data: 1
        size_mask: 1 X N
        dim_mask: 1 X D
        kernel_embeddings B_ens X D X N_k X d_k
        kernel_mask B_ens X D x N_k
        size_mask_kernel 1 x N x N
        kernel_list list of BaseKernelTypes - first list over ensemble batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
        """
        B_ens = len(kernel_list)
        assert B_ens == kernel_embeddings.shape[0] and B_ens == kernel_mask_enc.shape[0]
        dataset_embedding = self.dataset_encoder.forward(X_data, Y_data, size_mask, dim_mask, device)  # 1 X D X hidden_dim_1
        dataset_embedding_extended = dataset_embedding.repeat(B_ens, 1, 1)  # B_ens x D x hidden_dim_1
        dim_mask_extended = dim_mask.repeat(B_ens, 1)  # B_ens x D
        kernel_embeddings = self.kernel_encoder.forward(
            kernel_embeddings, dataset_embedding_extended, kernel_mask_enc, dim_mask_extended
        )  # B_ens X D X N_k x hidden_dim_2
        X_data_extended = X_data.repeat(B_ens, 1, 1)  # B_ens x N x D
        Y_data_extended = Y_data.repeat(B_ens, 1)  # B_ens x N
        size_mask_kernel_extended = size_mask_kernel.repeat(B_ens, 1, 1)  # B_ens x N x N
        size_mask_extended = size_mask.repeat(B_ens, 1)  # B_ens x N
        N_data_extended = N_data.repeat(B_ens)  # B_ens
        K = self.kernel_wrapper.forward(X_data_extended, X_data_extended, kernel_embeddings, kernel_list)  # B_ens x N x N
        log_prior_prob_kernel_params = self.kernel_wrapper.get_log_prior_prob(kernel_embeddings, kernel_list)  # B_ens
        noise_variances, _, log_prior_prob_variance = self.noise_variance_predictor.forward(
            kernel_embeddings, kernel_mask_enc, dataset_embedding_extended, dim_mask_extended
        )  # B_ens x 1, _, B_ens
        mll, mll_success = cal_marg_likelihood_batch_noise(
            K, Y_data_extended.unsqueeze(-1), noise_variances, size_mask_kernel_extended, 1.0 - size_mask_extended, N_data_extended, device
        )  # B
        # chol in mll might fail due to numerical instabilites - some upstream methods might not use the mll such as predict --> we dont throw an error here but push the information upstream
        if mll_success:
            nmll = -1.0 * mll
            nmll_with_prior = nmll - (log_prior_prob_kernel_params + log_prior_prob_variance) / N_data
        else:
            nmll = None
            nmll_with_prior = None

        return (
            kernel_embeddings,
            K,
            nmll,
            nmll_with_prior,
            noise_variances,
            log_prior_prob_kernel_params,
            log_prior_prob_variance,
            mll_success,
        )

    def warm_start(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        num_steps: int,
        learning_rate: float,
        device=torch.device("cpu"),
        only_learn_noise_variance: bool = False,
    ):
        """
        @TODO: right now only NMLL is optimized in warm start - not MAP
        X_data: B X N X D
        Y_data: B X N
        N_data: B
        dim_mask: B X D
        node_mask: B X N
        kernel_embeddings B X D X N_k X d_k
        kernel_mask B X D x N_k
        kernel_list list of BaseKernelTypes - first list over batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
        """
        assert self.kernel_wrapper.eval_mode == BaseKernelEvalMode.WARM_START
        dataset_embedding = self.dataset_encoder.forward(X_data, Y_data, size_mask, dim_mask, device)  # B X D X hidden_dim_1
        kernel_embeddings = self.kernel_encoder.forward(
            kernel_embeddings, dataset_embedding, kernel_mask_enc, dim_mask
        )  # B X D X N_k x hidden_dim_2
        untransformed_kernel_params = self.kernel_wrapper.get_untransformed_parameters(kernel_embeddings, kernel_list)
        _, noise_variances_untransformed, _ = self.noise_variance_predictor.forward(
            kernel_embeddings, kernel_mask_enc, dataset_embedding, dim_mask
        )  # B x 1
        noise_variances_untransformed = noise_variances_untransformed.clone().detach().requires_grad_(True)
        if only_learn_noise_variance:
            params = [noise_variances_untransformed]
        else:
            params = flatten_nested_list(untransformed_kernel_params) + [noise_variances_untransformed]
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        for i in range(0, num_steps):
            optimizer.zero_grad()
            K = self.kernel_wrapper.forward(X_data, X_data, kernel_embeddings, kernel_list, untransformed_kernel_params)  # B x N x N
            noise_variances = self.noise_variance_predictor.greater_than(
                noise_variances_untransformed, self.noise_variance_predictor.noise_variance_lower_bound
            )
            nmll = -cal_marg_likelihood_batch_noise(
                K, Y_data.unsqueeze(-1), noise_variances, size_mask_kernel, 1.0 - size_mask, N_data, device
            )[0]
            sum_nmll = torch.sum(nmll)
            sum_nmll.backward()
            optimizer.step()
        K = self.kernel_wrapper.forward(X_data, X_data, kernel_embeddings, kernel_list, untransformed_kernel_params)
        noise_variances = self.noise_variance_predictor.greater_than(
            noise_variances_untransformed, self.noise_variance_predictor.noise_variance_lower_bound
        )
        nmll = -cal_marg_likelihood_batch_noise(
            K, Y_data.unsqueeze(-1), noise_variances, size_mask_kernel, 1.0 - size_mask, N_data, device
        )[0]
        return untransformed_kernel_params, K, nmll, noise_variances


class ARDRBFOnlyAmortizedInferenceModel(BasicDimWiseAdditiveAmortizedInferenceModel):
    def __init__(
        self,
        dataset_encoder: DatasetEncoder,
        noise_variance_predictor: NoiseVariancePredictorHead,
        kernel_wrapper_hidden_layer_list: List[int],
        kernel_wrapper_dropout_p: float,
        kernel_wrapper_hidden_dim: int,
        eval_mode=BaseKernelEvalMode.STANDARD,
    ) -> None:
        DummyEncoder = namedtuple("DummyEncoder", "hidden_dim")
        super().__init__(
            dataset_encoder, DummyEncoder(hidden_dim=2), True, kernel_wrapper_hidden_layer_list, kernel_wrapper_dropout_p, eval_mode
        )
        self.eval_mode = eval_mode
        self.dataset_encoder = dataset_encoder
        self.kernel_wrapper = BatchedRBFKernelWrapper(
            kernel_wrapper_hidden_dim,
            kernel_wrapper_hidden_layer_list,
            kernel_wrapper_dropout_p,
            eval_mode=eval_mode,
        )
        self.noise_variance_predictor = noise_variance_predictor
        assert isinstance(self.noise_variance_predictor, NoiseVariancePredictorHead)
        self.noise_variance_predictor.set_eval_mode(self.eval_mode)
        self.warm_start_steps = None
        self.warm_start_lr = None

    def set_eval_mode(self, eval_mode: BaseKernelEvalMode):
        super().set_eval_mode(eval_mode)
        self.noise_variance_predictor.set_eval_mode(self.eval_mode)

    def forward(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        X_data: B X N X D
        Y_data: B X N
        N_data: B
        size_mask: B X N
        dim_mask: B X D
        kernel_embeddings B X D X N_k X d_k
        kernel_mask_enc B X D x N_k
        size_mask_kernel B x N x N
        kernel_list list of BaseKernelTypes - first list over batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
        """
        dataset_embedding = self.dataset_encoder.forward(X_data, Y_data, size_mask, dim_mask, device)  # B X D X kernel_wrapper_hidden_dim
        # overwrite kernel embeddings and kernel mask to align with RBF kernel instead of kernel_list
        kernel_embeddings = dataset_embedding.unsqueeze(-2)  # B X D X 1 x kernel_wrapper_hidden_dim
        kernel_mask_enc = dim_mask.unsqueeze(-1)  # B X D x 1
        K = self.kernel_wrapper.forward(X_data, X_data, kernel_embeddings, kernel_list)  # B x N x N
        log_prior_prob_kernel_params = self.kernel_wrapper.get_log_prior_prob(kernel_embeddings, kernel_list)  # B
        noise_variances, _, log_prior_prob_variance = self.noise_variance_predictor.forward(
            kernel_embeddings, kernel_mask_enc, dataset_embedding, dim_mask
        )  # B x 1, _, B
        mll, mll_success = cal_marg_likelihood_batch_noise(
            K, Y_data.unsqueeze(-1), noise_variances, size_mask_kernel, 1.0 - size_mask, N_data, device
        )  # B
        # chol in mll might fail due to numerical instabilites - some upstream methods might not use the mll such as predict --> we dont throw an error here but push the information upstream
        if mll_success:
            nmll = -1.0 * mll
            nmll_with_prior = nmll - (log_prior_prob_kernel_params + log_prior_prob_variance) / N_data
        else:
            nmll = None
            nmll_with_prior = None

        return (
            kernel_embeddings,
            K,
            nmll,
            nmll_with_prior,
            noise_variances,
            log_prior_prob_kernel_params,
            log_prior_prob_variance,
            mll_success,
        )

    def forward_ensemble(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        device=torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        raise NotImplementedError

    def warm_start(
        self,
        X_data: torch.Tensor,
        Y_data: torch.Tensor,
        N_data: torch.Tensor,
        size_mask: torch.Tensor,
        dim_mask: torch.Tensor,
        kernel_embeddings: torch.Tensor,
        kernel_mask_enc: torch.Tensor,
        size_mask_kernel: torch.Tensor,
        kernel_list: List[List[List[BaseKernelTypes]]],
        num_steps: int,
        learning_rate: float,
        device=torch.device("cpu"),
        only_learn_noise_variance: bool = False,
    ):
        raise NotImplementedError

    def get_num_predicted_params(self, kernel_list: List[List[List[BaseKernelTypes]]], device=torch.device("cpu")):
        n_kernel_params = self.kernel_wrapper.get_num_params(kernel_list)
        n_kernel_params = torch.Tensor(n_kernel_params).to(device)  # B
        n_gp_params = n_kernel_params + 1.0
        return n_gp_params


def to_input_tensor_of_dim_wise_additive(kernel_list, device, X_list, y_list):
    """
    y_list - list over (1d np.array) of potential different lenght (n_i,)
    X_list - list over np.arrays (of potential different shapes (n_i,d_i))
    kernel_list list of BaseKernelTypes - first list over batch - second list over dimension - third list over kernel symbols e.g [[[SE,PER,LIN],[SE_MULT_PER,SE]],[[SE,LIN],[SE]]]
    """
    kernel_embeddings, kernel_mask = get_kernel_embeddings_and_kernel_mask(kernel_list)
    X_padded, y_padded, size_mask, dim_mask, N, size_mask_kernel = get_padded_dataset_and_masks(X_list, y_list)
    kernel_embeddings = torch.from_numpy(kernel_embeddings).float().to(device)
    kernel_mask = torch.from_numpy(kernel_mask).float().to(device)
    X_padded = torch.from_numpy(X_padded).float().to(device)
    y_padded = torch.from_numpy(y_padded).float().to(device)
    size_mask = torch.from_numpy(size_mask).float().to(device)
    dim_mask = torch.from_numpy(dim_mask).float().to(device)
    size_mask_kernel = torch.from_numpy(size_mask_kernel).float().to(device)
    N = torch.from_numpy(N).float().to(device)
    return kernel_mask, kernel_embeddings, X_padded, y_padded, size_mask, dim_mask, N, size_mask_kernel


if __name__ == "__main__":
    pass
