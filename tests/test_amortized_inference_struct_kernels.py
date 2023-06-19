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
import numpy as np
import torch
from torch.utils.data import DataLoader
from amorstructgp.config.kernels.gpytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicRBFPytorchConfig,
    BasicLinearKernelPytorchConfig,
    BasicPeriodicKernelPytorchConfig,
    BasicMatern52PytorchConfig,
    BasicRQKernelPytorchConfig,
)
from amorstructgp.config.models.gp_model_gpytorch_config import BasicGPModelPytorchConfig
from amorstructgp.gp.gpytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from amorstructgp.utils.enums import PredictionQuantity
from amorstructgp.gp.kernel_grammar import (
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarExpressionTransformer,
    KernelGrammarOperator,
)
from amorstructgp.config.data_generators.dim_wise_additive_generator_config import (
    DimWiseAdditiveWithNoiseMixedNoMaternConfig,
)
from amorstructgp.config.models.gp_model_gpytorch_config import BasicGPModelPytorchConfig
from amorstructgp.gp.gpytorch_kernels.elementary_kernels_pytorch import (
    RBFKernelPytorch,
    LinearKernelPytorch,
    RQKernelPytorch,
    PeriodicKernelPytorch,
    Matern52KernelPytorch,
)
from amorstructgp.data_generators.dataset_of_datasets import RandomDatasetOfDatasets
from amorstructgp.data_generators.generator_factory import GeneratorFactory
from amorstructgp.data_generators.simulator import Simulator
from amorstructgp.gp.utils import cal_marg_likelihood, cal_marg_likelihood_batch_noise
from amorstructgp.nn.amortized_inference_models import (
    DimWiseAdditiveKernelWithNoiseAmortizedModel,
    DimWiseAdditiveKernelsAmortizedInferenceModel,
)
from amorstructgp.nn.amortized_models_factory import AmortizedModelsFactory
from amorstructgp.config.nn.amortized_infer_models_configs import (
    ARDRBFOnlyAmortizedModelConfig,
    BiggerARDRBFOnlyAmortizedModelConfig,
    DimWiseAdditiveKernelWithNoiseAmortizedModelConfig,
    SharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig,
    DimWiseAdditiveWithNoiseUnsharedWeightsAdditionAmortizedModelConfig,
    FullDimGlobalNoiseDimWiseAdditiveAmortizedModelConfig,
    SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    WiderCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    WiderStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
)
from amorstructgp.nn.dataset_encoder import (
    DatasetEncoder,
    EnrichedDatasetEncoder,
    FullDimDatasetEncoder,
    get_padded_dataset_and_masks,
)
from amorstructgp.nn.kernel_encoder import CrossAttentionKernelEncoder, KernelEncoder
from amorstructgp.gp.base_kernels import (
    BaseKernelEvalMode,
    BatchedDimWiseAdditiveKernelWrapper,
    DimWiseAdditiveKernelWrapper,
    LinearKernel,
    PeriodicKernel,
    get_batch_from_nested_parameter_list,
    get_expressions_from_kernel_list,
    get_kernel_embeddings_and_kernel_mask,
    BaseKernelTypes,
    get_parameter_value_lists,
    transform_kernel_list_to_expression,
    MaternKernel,
)
from amorstructgp.gp.base_kernels import SEKernel
import pytest
import random
from amorstructgp.nn.noise_predictor_head import NoiseVariancePredictorHead
from amorstructgp.models.gp_model_amortized_structured import GPModelAmortizedStructured
from amorstructgp.models.gp_model_pytorch import GPModelPytorch
from amorstructgp.models.model_factory import ModelFactory
import copy
from amorstructgp.utils.utils import (
    default_worker_init_function,
    get_number_of_parameters,
    get_test_inputs,
)
import gpytorch
from amorstructgp.gp.base_kernels import N_BASE_KERNEL_TYPES
from amorstructgp.utils.gpytorch_utils import (
    get_gpytorch_kernel_from_expression_and_state_dict,
    get_hp_sample_from_prior_gpytorch_as_state_dict,
)
from amorstructgp.config.prior_parameters import NOISE_VARIANCE_EXPONENTIAL_LAMBDA
import time


@pytest.mark.parametrize("max_dim,min_dim,n_max,n_min,B", [(5, 2, 4, 2, 6), (7, 4, 5, 3, 3), (5, 3, 5, 1, 10)])
def test_kernel_encoding_masking(max_dim, min_dim, n_max, n_min, B):
    kernel_types = [
        BaseKernelTypes.LIN,
        BaseKernelTypes.SE,
        BaseKernelTypes.PER,
        BaseKernelTypes.SE_MULT_PER,
        BaseKernelTypes.LIN_MULT_PER,
        BaseKernelTypes.SE_MULT_LIN,
        BaseKernelTypes.SE_MULT_MATERN52,
    ]
    kernel_list = []
    i_min, i_max = np.random.choice(list(range(0, B)), 2, replace=False)
    for i in range(0, B):
        if i == i_min:
            n = n_min
            D = min_dim
        elif i == i_max:
            n = n_max
            D = max_dim
        else:
            n = np.random.randint(n_min, n_max)
            D = np.random.randint(min_dim, max_dim)
        kernel_list_i = []
        for d in range(0, D):
            kernel_list_i_d = np.random.choice(kernel_types, n, replace=True)
            kernel_list_i.append(kernel_list_i_d)
        kernel_list.append(kernel_list_i)
    kernel_embeddings, kernel_mask = get_kernel_embeddings_and_kernel_mask(kernel_list)
    assert np.allclose(np.sum(kernel_mask[i_min, min_dim:, :]), 0.0)
    assert np.allclose(np.sum(kernel_mask[i_min, :, n_min:]), 0.0)
    assert np.allclose(kernel_mask[i_max], 1.0)
    assert np.allclose(kernel_embeddings[i_min, min_dim:, :, :], 0.0)
    assert np.allclose(kernel_embeddings[i_min, :, n_min:, :], 0.0)


@pytest.mark.parametrize("d,n_train,n_test", [(2, 20, 10)])
def test_gp_model_amortized_structured_wrapper(d, n_train, n_test):
    X_test = np.random.uniform(0.0, 1.0, size=(n_test, d))
    X_data = np.random.uniform(0.0, 1.0, size=(n_train, d))
    y_data = np.random.uniform(0.0, 1.0, size=(n_train, 1))
    kernel_list = [[BaseKernelTypes.SE] for i in range(0, d)]
    model_config = WiderCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig()
    model = AmortizedModelsFactory.build(model_config)
    state_dict = copy.deepcopy(model.state_dict())
    wrapper_model = GPModelAmortizedStructured(PredictionQuantity.PREDICT_Y, model_config, False, 10, 0.1)
    wrapper_model.build_model(state_dict)
    wrapper_model.set_kernel_list(kernel_list)
    wrapper_model.infer(X_data, y_data)
    mu_test_wrapper, sigma_test_wrapper = wrapper_model.predictive_dist(X_test)
    model.eval()
    mu_test, _, sigma_test = model.predict(X_test, X_data, y_data, kernel_list)
    assert np.allclose(mu_test, mu_test_wrapper, atol=1e-5)
    assert np.allclose(sigma_test, sigma_test_wrapper, atol=1e-5)


@pytest.mark.parametrize("d,n_train,n_test", [(2, 20, 10)])
def test_predict_amortized_model(d, n_train, n_test):
    dataset_encoder = DatasetEncoder(3, 3, 12, 12, 0.1)
    kernel_encoder = KernelEncoder(N_BASE_KERNEL_TYPES, 3, 12, 12, 0.1)
    X_test = np.random.uniform(0.0, 1.0, size=(n_test, d))
    X_data = np.random.uniform(0.0, 1.0, size=(n_train, d))
    y_data = np.random.uniform(0.0, 1.0, size=(n_train, 1))
    model = DimWiseAdditiveKernelsAmortizedInferenceModel(dataset_encoder, kernel_encoder, True, [], 0.1, 1e-2, BaseKernelEvalMode.DEBUG)
    kernel_list = [[BaseKernelTypes.SE] for i in range(0, d)]
    mu_test, _, sigma_test = model.predict(X_test, X_data, y_data, kernel_list)
    kernel_config = BasicRBFPytorchConfig(input_dimension=d)
    model_config = BasicGPModelPytorchConfig(kernel_config=kernel_config)
    model_config.optimize_hps = False
    model_config.initial_likelihood_noise = 0.1
    model = ModelFactory.build(model_config)
    # model.kernel = gpflow.kernels.RBF()
    model.infer(X_data, y_data)
    mu_gpytorch, sigma_gpytorch = model.predictive_dist(X_test)
    assert np.allclose(mu_test, mu_gpytorch, atol=1e-5)
    assert np.allclose(sigma_test, sigma_gpytorch, atol=1e-5)


@pytest.mark.parametrize("d,n_train,n_test", [(2, 20, 10)])
def test_predict_rbf_amortized_model(d, n_train, n_test):
    X_test = np.random.uniform(0.0, 1.0, size=(n_test, d))
    X_data = np.random.uniform(0.0, 1.0, size=(n_train, d))
    y_data = np.random.uniform(0.0, 1.0, size=(n_train, 1))
    model = AmortizedModelsFactory.build(ARDRBFOnlyAmortizedModelConfig())
    model.set_eval_mode(BaseKernelEvalMode.DEBUG)
    kernel_list = [[BaseKernelTypes.SE, BaseKernelTypes.SE_MULT_LIN] for i in range(0, d)]
    mu_test, _, sigma_test = model.predict(X_test, X_data, y_data, kernel_list)
    kernel_config = BasicRBFPytorchConfig(input_dimension=d)
    model_config = BasicGPModelPytorchConfig(kernel_config=kernel_config)
    model_config.optimize_hps = False
    model_config.initial_likelihood_noise = 0.1
    model = ModelFactory.build(model_config)
    # model.kernel = gpflow.kernels.RBF()
    model.infer(X_data, y_data)
    mu_gpytorch, sigma_gpytorch = model.predictive_dist(X_test)
    assert np.allclose(mu_test, mu_gpytorch, atol=1e-4)
    assert np.allclose(sigma_test, sigma_gpytorch, atol=1e-4)


@pytest.mark.parametrize(
    "d,n_train,n_test,base_kernel_type",
    [
        (2, 20, 10, BaseKernelTypes.SE),
        (2, 20, 10, BaseKernelTypes.PER),
        (4, 20, 10, BaseKernelTypes.LIN_MULT_PER),
        (4, 20, 10, BaseKernelTypes.LIN_MULT_MATERN52),
    ],
)
def test_predict_amortized_model_warm_start(d, n_train, n_test, base_kernel_type):
    dataset_encoder = DatasetEncoder(3, 3, 12, 12, 0.1)
    kernel_encoder = KernelEncoder(N_BASE_KERNEL_TYPES, 3, 12, 12, 0.1)
    noise_predictor = NoiseVariancePredictorHead([20], 12, 1e-4, 0.2, False)
    X_test = np.random.uniform(0.0, 1.0, size=(n_test, d))
    X_data = np.random.uniform(0.0, 1.0, size=(n_train, d))
    y_data = np.random.uniform(0.0, 1.0, size=(n_train, 1))
    model = DimWiseAdditiveKernelWithNoiseAmortizedModel(
        dataset_encoder, kernel_encoder, noise_predictor, True, [], 0.1, BaseKernelEvalMode.STANDARD
    )
    model.eval()
    kernel_list = [[base_kernel_type] for i in range(0, d)]
    mu_test, _, sigma_test = model.predict(X_test, X_data, y_data, kernel_list)
    model.set_eval_mode(BaseKernelEvalMode.WARM_START)
    model.set_warm_start_params(0, 0.01)
    mu_test2, _, sigma_test2 = model.predict(X_test, X_data, y_data, kernel_list)
    assert np.allclose(mu_test, mu_test2, rtol=1e-4, atol=1e-5)
    assert np.allclose(sigma_test, sigma_test2, atol=1e-5)
    model.set_warm_start_params(20, 0.01)
    mu_test3, _, sigma_test3 = model.predict(X_test, X_data, y_data, kernel_list)
    assert not np.allclose(mu_test3, mu_test2)
    assert not np.allclose(sigma_test3, sigma_test2)


@pytest.mark.parametrize(
    "model_config",
    [
        DimWiseAdditiveKernelWithNoiseAmortizedModelConfig(),
        SharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig(),
        DimWiseAdditiveWithNoiseUnsharedWeightsAdditionAmortizedModelConfig(),
        FullDimGlobalNoiseDimWiseAdditiveAmortizedModelConfig(),
        WiderStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig(),
        ARDRBFOnlyAmortizedModelConfig(),
    ],
)
def test_eval_amortized_model(model_config):
    B = 5
    (
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
        kernel_list,
    ) = get_test_inputs(B, 20, 4, 3, only_SE=False)
    model = AmortizedModelsFactory.build(model_config)
    print(get_number_of_parameters(model))
    for i in range(0, 1):
        before = time.perf_counter()
        model.forward(X_padded, y_padded, N, size_mask, dim_mask, kernel_embeddings, kernel_mask, size_mask_kernel, kernel_list)
        print(kernel_list)
        model.predict(X_list[0], X_list[0], np.expand_dims(y_list[0], axis=1), kernel_list[0])
        after = time.perf_counter()
        difference = after - before
        print(difference)


def test_input_equivariance_kernel_encoding():
    B = 5
    X_list, y_list, X_padded, y_padded, N, size_mask, dim_mask, _, _, size_mask_kernel, kernel_list = get_test_inputs(
        B, 10, 4, 1, only_SE=True
    )
    new_kernel_list_single_dimensions = [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_MATERN52]
    new_kernel_list_single_dimensions_permuted = [BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_MATERN52, BaseKernelTypes.SE]
    kernel_list1 = copy.deepcopy(kernel_list)
    kernel_list2 = copy.deepcopy(kernel_list)
    kernel_list1[0][0] = new_kernel_list_single_dimensions
    kernel_list2[0][0] = new_kernel_list_single_dimensions_permuted
    kernel_embeddings1, kernel_mask_1 = get_kernel_embeddings_and_kernel_mask(kernel_list1)
    kernel_embeddings2, kernel_mask_2 = get_kernel_embeddings_and_kernel_mask(kernel_list2)
    kernel_embeddings1 = torch.from_numpy(kernel_embeddings1).float()
    kernel_embeddings2 = torch.from_numpy(kernel_embeddings2).float()
    kernel_mask_1 = torch.from_numpy(kernel_mask_1).float()
    kernel_mask_2 = torch.from_numpy(kernel_mask_2).float()
    permutation_indexes = [2, 0, 1]
    dataset_encoder = DatasetEncoder(3, 3, 12, 12, 0.1)
    kernel_encoder_list = [
        KernelEncoder(N_BASE_KERNEL_TYPES, 3, 12, 12, 0.1),
        CrossAttentionKernelEncoder(N_BASE_KERNEL_TYPES, 12, 12, 3, 3, 3, False, 0.1),
        CrossAttentionKernelEncoder(N_BASE_KERNEL_TYPES, 12, 12, 3, 3, 3, True, 0.1),
    ]
    for kernel_encoder in kernel_encoder_list:
        dataset_encoding = dataset_encoder.forward(X_padded, y_padded, size_mask, dim_mask)
        kernel_encoder.eval()
        kernel_encoding_1 = kernel_encoder.forward(kernel_embeddings1, dataset_encoding, kernel_mask_1, dim_mask).detach().numpy()
        kernel_encoding_2 = kernel_encoder.forward(kernel_embeddings2, dataset_encoding, kernel_mask_2, dim_mask).detach().numpy()
        for index in range(0, len(permutation_indexes)):
            assert np.allclose(
                kernel_encoding_1[0, 0, index, :], kernel_encoding_2[0, 0, permutation_indexes[index], :], rtol=1e-4, atol=1e-6
            )


def test_kernel_list_to_expression_transformation():
    kernel_list = [[BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.LIN], [BaseKernelTypes.SE_MULT_PER, BaseKernelTypes.PER]]
    expression = transform_kernel_list_to_expression(kernel_list, True)
    kernel = expression.get_kernel()
    assert expression.count_operators() == 5
    assert expression.count_elementary_expressions() == 6
    X = np.random.uniform(0.0, 1.0, (20, 2))
    X_torch = torch.from_numpy(X)
    K = kernel(X_torch)
    kernel2 = gpytorch.kernels.AdditiveKernel(kernel)
    state_dict = kernel2.state_dict()
    kernel4 = get_gpytorch_kernel_from_expression_and_state_dict(expression, state_dict, True)
    K2 = kernel4(X_torch)
    assert np.allclose(K.numpy(), K2.numpy())


def test_kernel_list_to_expression_transformation2():
    kernel_list = [
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.LIN_MULT_PER, BaseKernelTypes.SE],
        [BaseKernelTypes.LIN],
    ]

    se_on_0 = ElementaryKernelGrammarExpression(
        RBFKernelPytorch(**BasicRBFPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict())
    )
    lin_on_0 = ElementaryKernelGrammarExpression(
        LinearKernelPytorch(**BasicLinearKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict())
    )
    per_on_0 = ElementaryKernelGrammarExpression(
        PeriodicKernelPytorch(
            **BasicPeriodicKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict()
        )
    )
    se_on_1 = ElementaryKernelGrammarExpression(
        RBFKernelPytorch(**BasicRBFPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict())
    )
    lin_on_1 = ElementaryKernelGrammarExpression(
        LinearKernelPytorch(**BasicLinearKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict())
    )
    per_on_1 = ElementaryKernelGrammarExpression(
        PeriodicKernelPytorch(
            **BasicPeriodicKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict()
        )
    )
    lin_on_2 = ElementaryKernelGrammarExpression(
        LinearKernelPytorch(**BasicLinearKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=2).dict())
    )
    lin_mult_per_on_0 = KernelGrammarExpression(lin_on_0, per_on_0, KernelGrammarOperator.MULTIPLY)
    lin_mult_per_on_1 = KernelGrammarExpression(lin_on_1, per_on_1, KernelGrammarOperator.MULTIPLY)
    expression_dim_0 = KernelGrammarExpression(
        KernelGrammarExpression(se_on_0, lin_on_0, KernelGrammarOperator.ADD), lin_mult_per_on_0, KernelGrammarOperator.ADD
    )
    expression_dim_1 = KernelGrammarExpression(lin_mult_per_on_1, se_on_1, KernelGrammarOperator.ADD)
    expression_dim_1_and_2 = KernelGrammarExpression(expression_dim_1, lin_on_2, KernelGrammarOperator.MULTIPLY)
    expression = KernelGrammarExpression(expression_dim_0, expression_dim_1_and_2, KernelGrammarOperator.MULTIPLY)
    expression2 = transform_kernel_list_to_expression(kernel_list)
    X = np.random.rand(20, 3)
    X2 = np.random.rand(10, 3)
    X_torch = torch.from_numpy(X)
    X_torch2 = torch.from_numpy(X2)
    kernel = expression.get_kernel()
    kernel2 = expression2.get_kernel()
    K1 = kernel(X_torch, X_torch2).numpy()
    K2 = kernel2(X_torch, X_torch2).numpy()
    assert np.allclose(K1, K2)


def test_input_invariance_dataset_encoding():
    dataset_encoder_list = [
        DatasetEncoder(3, 3, 8, 8, 0.1),
        EnrichedDatasetEncoder(3, 3, 3, 3, 8, True, 10, 0.1),
        EnrichedDatasetEncoder(3, 3, 3, 3, 8, False, 10, 0.1),
    ]
    for dataset_encoder in dataset_encoder_list:
        x1 = np.random.randn(1, 5)
        x2 = np.random.randn(1, 5)
        x3 = np.random.randn(1, 5)
        x4 = np.random.randn(1, 5)
        y1 = np.random.randn(1, 1)
        y2 = np.random.randn(1, 1)
        y3 = np.random.randn(1, 1)
        y4 = np.random.randn(1, 1)
        X = np.concatenate((x1, x2, x3, x4))
        y = np.concatenate((y1, y2, y3, y4))
        y = np.squeeze(y)
        X_perm = np.concatenate((x2, x1, x4, x3))
        y_perm = np.concatenate((y2, y1, y4, y3))
        y_perm = np.squeeze(y_perm)
        X_padded, y_padded, size_mask, dim_mask, _, _ = get_padded_dataset_and_masks([X], [y])
        X_padded_perm, y_padded_perm, size_mask_perm, dim_mask_perm, _, _ = get_padded_dataset_and_masks([X_perm], [y_perm])
        # dataset_encoder = DatasetEncoder(3, 3, 8, 8)
        dataset_encoder.eval()
        output = dataset_encoder.forward(
            torch.from_numpy(X_padded).float(),
            torch.from_numpy(y_padded).float(),
            torch.from_numpy(size_mask).float(),
            torch.from_numpy(dim_mask).float(),
        )
        output_perm = dataset_encoder.forward(
            torch.from_numpy(X_padded_perm).float(),
            torch.from_numpy(y_padded_perm).float(),
            torch.from_numpy(size_mask_perm).float(),
            torch.from_numpy(dim_mask_perm).float(),
        )
        assert np.allclose(output.detach().numpy(), output_perm.detach().numpy(), rtol=1e-4, atol=1e-6)
        X_dim0 = X[:, 0]
        X_dim2 = X[:, 2]
        X_dim_permute = X.copy()
        X_dim_permute[:, 0] = X_dim2
        X_dim_permute[:, 2] = X_dim0
        X_padded_dim_perm, y_padded_dim_perm, size_mask_dim_perm, dim_mask_dim_perm, _, _ = get_padded_dataset_and_masks(
            [X_dim_permute], [y]
        )
        output_dim_perm = dataset_encoder.forward(
            torch.from_numpy(X_padded_dim_perm).float(),
            torch.from_numpy(y_padded_dim_perm).float(),
            torch.from_numpy(size_mask_dim_perm).float(),
            torch.from_numpy(dim_mask_dim_perm).float(),
        )
        output_dim_perm = output_dim_perm.detach().numpy()
        out_dim_0 = output_dim_perm[:, 0, :]
        out_dim_2 = output_dim_perm[:, 2, :]
        output_dim_perm_perm = output_dim_perm.copy()
        output_dim_perm_perm[:, 2, :] = out_dim_0
        output_dim_perm_perm[:, 0, :] = out_dim_2
        assert np.allclose(output_dim_perm_perm, output.detach().numpy(), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("sample_noise", (True, False))
def test_simulated_dataset_consistency(sample_noise: bool):
    kernel_list = [[BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.LIN], [BaseKernelTypes.SE_MULT_PER, BaseKernelTypes.PER]]
    expression = transform_kernel_list_to_expression(kernel_list, True)
    simulator = Simulator(0.0, 1.0, NOISE_VARIANCE_EXPONENTIAL_LAMBDA)
    simulated_dataset_object = simulator.create_sample(40, 50, expression, 0.1, sample_noise)
    X, y = simulated_dataset_object.get_dataset()
    kernel = get_gpytorch_kernel_from_expression_and_state_dict(
        simulated_dataset_object.get_kernel_expression_gt(), simulated_dataset_object.get_kernel_state_dict_gt(), True
    )
    X_test, y_test = simulated_dataset_object.get_test_dataset()
    assert len(X_test) == 50
    assert len(y_test) == 50
    X_torch = torch.from_numpy(X)
    y_torch = torch.squeeze(torch.from_numpy(y))
    model_config = BasicGPModelPytorchConfig(kernel_config=BasicRBFPytorchConfig(input_dimension=0))
    model_config.add_constant_mean_function = False
    model_config.fix_likelihood_variance = True
    model_config.optimize_hps = False
    model_config.observation_noise_variance_lambda = simulated_dataset_object.noise_variance_lambda
    model_config.initial_likelihood_noise = simulated_dataset_object.observation_noise
    if sample_noise:
        model_config.set_prior_on_observation_noise = True
    else:
        model_config.set_prior_on_observation_noise = False
    model = GPModelPytorch(kernel=kernel, **model_config.dict())
    model.infer(X, y)
    mll = model.eval_log_marginal_likelihood(X, y)

    assert np.allclose(mll.detach().numpy(), simulated_dataset_object.mll_gt)


def test_dim_wise_additive_kernel_wrapper2():
    all_kernel_symbols = [kernel_type for kernel_type in BaseKernelTypes]
    dim = 4
    kernel_len = 3
    n_repeats = 10
    X = np.random.rand(20, dim)
    X2 = np.random.rand(10, dim)
    X_torch = torch.from_numpy(X)
    X_torch2 = torch.from_numpy(X2)
    additive_wrapper = DimWiseAdditiveKernelWrapper(N_BASE_KERNEL_TYPES, [], 0.1, True, BaseKernelEvalMode.DEBUG)
    for i in range(0, n_repeats):
        kernel_list = []
        for d in range(0, dim):
            kernel_list_in_dim = []
            for j in range(0, kernel_len):
                index = np.random.randint(0, len(all_kernel_symbols) - 1)
                kernel_list_in_dim.append(all_kernel_symbols[index])
            kernel_list.append(kernel_list_in_dim)
        kernel_expression = transform_kernel_list_to_expression(kernel_list)
        kernel_embeddings, kernel_mask = get_kernel_embeddings_and_kernel_mask([kernel_list])
        kernel_embeddings = kernel_embeddings[0]
        for d in range(0, dim):
            for j in range(0, kernel_len):
                embedding = np.zeros(10)
                embedding[int(kernel_list[d][j])] = 1.0
                assert np.allclose(kernel_embeddings[d, j, :], embedding)
        K_torch = additive_wrapper.forward(X_torch, X_torch2, kernel_embeddings, kernel_list)
        K_gpytorch = kernel_expression.get_kernel()(X_torch, X_torch2).detach().numpy()
        assert np.allclose(K_torch.numpy(), K_gpytorch)


def test_dim_wise_additive_kernel_wrapper():
    dim = 3
    kernel_list = [
        [BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.LIN_MULT_PER],
        [BaseKernelTypes.LIN_MULT_PER, BaseKernelTypes.SE],
        [BaseKernelTypes.LIN],
    ]

    se_on_0 = ElementaryKernelGrammarExpression(
        RBFKernelPytorch(**BasicRBFPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict())
    )
    lin_on_0 = ElementaryKernelGrammarExpression(
        LinearKernelPytorch(**BasicLinearKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict())
    )
    per_on_0 = ElementaryKernelGrammarExpression(
        PeriodicKernelPytorch(
            **BasicPeriodicKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict()
        )
    )
    se_on_1 = ElementaryKernelGrammarExpression(
        RBFKernelPytorch(**BasicRBFPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict())
    )
    lin_on_1 = ElementaryKernelGrammarExpression(
        LinearKernelPytorch(**BasicLinearKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict())
    )
    per_on_1 = ElementaryKernelGrammarExpression(
        PeriodicKernelPytorch(
            **BasicPeriodicKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict()
        )
    )
    lin_on_2 = ElementaryKernelGrammarExpression(
        LinearKernelPytorch(**BasicLinearKernelPytorchConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=2).dict())
    )
    lin_mult_per_on_0 = KernelGrammarExpression(lin_on_0, per_on_0, KernelGrammarOperator.MULTIPLY)
    lin_mult_per_on_1 = KernelGrammarExpression(lin_on_1, per_on_1, KernelGrammarOperator.MULTIPLY)
    expression_dim_0 = KernelGrammarExpression(
        KernelGrammarExpression(se_on_0, lin_on_0, KernelGrammarOperator.ADD), lin_mult_per_on_0, KernelGrammarOperator.ADD
    )
    expression_dim_1 = KernelGrammarExpression(lin_mult_per_on_1, se_on_1, KernelGrammarOperator.ADD)
    expression_dim_1_and_2 = KernelGrammarExpression(expression_dim_1, lin_on_2, KernelGrammarOperator.MULTIPLY)
    expression = KernelGrammarExpression(expression_dim_0, expression_dim_1_and_2, KernelGrammarOperator.MULTIPLY)
    kernel_embeddings, kernel_mask = get_kernel_embeddings_and_kernel_mask([kernel_list])
    kernel_embeddings = kernel_embeddings[0]

    X = np.random.rand(20, dim)
    X2 = np.random.rand(10, dim)
    X_torch = torch.from_numpy(X)
    X_torch2 = torch.from_numpy(X2)
    additive_wrapper = DimWiseAdditiveKernelWrapper(N_BASE_KERNEL_TYPES, [], 0.1, True, BaseKernelEvalMode.DEBUG)
    K_torch = additive_wrapper.forward(X_torch, X_torch2, kernel_embeddings, kernel_list)
    K_gpytorch = expression.get_kernel()(X_torch, X_torch2).detach().numpy()
    assert np.allclose(K_torch.numpy(), K_gpytorch)


def test_marginal_likelihood():
    X_list, y_list, X_padded, y_padded, N, size_mask, _, kernel_embeddings, _, size_mask_kernel, kernel_list = get_test_inputs(
        5, 10, 4, 1, only_SE=True
    )
    kernel_wrapper = BatchedDimWiseAdditiveKernelWrapper(4, [], 0.1, True, BaseKernelEvalMode.DEBUG)
    K = kernel_wrapper.forward(X_padded, X_padded, kernel_embeddings, kernel_list)
    mll = cal_marg_likelihood(K, y_padded.unsqueeze(-1), 1e-2, size_mask_kernel, 1.0 - size_mask, N, torch.device("cpu"))[0] * N
    mlls_gpytorch = []
    for i, x_data in enumerate(X_list):
        y_data = y_list[i]
        n = x_data.shape[0]
        d = x_data.shape[1]
        kernel_config = BasicRBFPytorchConfig(input_dimension=d)
        model_config = BasicGPModelPytorchConfig(kernel_config=kernel_config)
        model_config.optimize_hps = False
        model_config.initial_likelihood_noise = 0.1
        model = ModelFactory.build(model_config)
        # model.kernel = gpflow.kernels.RBF()
        model.infer(x_data, np.expand_dims(y_data, axis=1))
        mll_i = model.eval_log_marginal_likelihood(x_data, np.squeeze(y_data)).detach().numpy() * n
        mlls_gpytorch.append(mll_i)
    assert np.allclose(np.array(mlls_gpytorch), mll, atol=1e-4, rtol=1e-4)


def test_kernel_expression_kernel_list_alignment():
    kernel_list = [[BaseKernelTypes.SE, BaseKernelTypes.LIN, BaseKernelTypes.LIN], [BaseKernelTypes.SE_MULT_PER, BaseKernelTypes.PER]]
    combined_expression, single_expression_list = get_expressions_from_kernel_list(kernel_list, make_deep_copy=False)
    kernel_state_dict = get_hp_sample_from_prior_gpytorch_as_state_dict(combined_expression, wrap_in_addition=True)
    kernel_expression_list = []
    for d in range(0, len(single_expression_list)):
        kernel_expression_list_dimension = []
        for expression in single_expression_list[d]:
            kernel_expression_list_dimension.append(expression)

        expression_for_dimension = KernelGrammarExpressionTransformer.add_expressions(
            kernel_expression_list_dimension, make_deep_copy=False
        )
        kernel_expression_list.append(expression_for_dimension)
    combined_expression_2 = KernelGrammarExpressionTransformer.multiply_expressions(kernel_expression_list, make_deep_copy=False)
    kernel_3 = get_gpytorch_kernel_from_expression_and_state_dict(combined_expression, kernel_state_dict, wrap_in_addition=True)
    kernel_1 = combined_expression.get_kernel()
    kernel_2 = combined_expression_2.get_kernel()
    X = torch.from_numpy(np.random.rand(10, len(kernel_list)))
    K3 = kernel_3(X, X).numpy()
    K1 = kernel_1(X, X).numpy()
    K2 = kernel_2(X, X).numpy()
    assert np.allclose(K3, K1)
    assert np.allclose(K2, K1)


def test_marginal_likelihood_extented():
    B = 5
    (
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
        kernel_list,
    ) = get_test_inputs(B, 10, 4, 2, only_SE=False)
    model = AmortizedModelsFactory.build(DimWiseAdditiveKernelWithNoiseAmortizedModelConfig())
    model.set_eval_mode(BaseKernelEvalMode.DEBUG)
    model.eval()
    kernel_embeddings, K, nmll, nmll_with_prior, noise_variances, _, _, mll_success = model.forward(
        X_padded, y_padded, N, size_mask, dim_mask, kernel_embeddings, kernel_mask, size_mask_kernel, kernel_list
    )
    num_params = model.get_num_predicted_params(kernel_list)
    mll = -1 * nmll * N
    mll_with_prior = -1 * nmll_with_prior * N
    mlls_gpytorch = []
    mlls_with_prior_gpytorch = []
    for i, x_data in enumerate(X_list):
        y_data = y_list[i]
        n = x_data.shape[0]
        d = x_data.shape[1]
        kernel_list_element = kernel_list[i]
        kernel_expression = transform_kernel_list_to_expression(kernel_list_element, True, True)
        kernel_config = BasicRBFPytorchConfig(input_dimension=d)
        model_config = BasicGPModelPytorchConfig(kernel_config=kernel_config)
        model_config.optimize_hps = False
        model_config.set_prior_on_observation_noise = True
        model_config.initial_likelihood_noise = 0.1
        model = ModelFactory.build(model_config)
        model.kernel_module = kernel_expression.get_kernel()
        # model.kernel = gpflow.kernels.RBF()
        model.infer(x_data, np.expand_dims(y_data, axis=1))
        mll_i = model.eval_log_marginal_likelihood(x_data, np.squeeze(y_data)).detach().numpy() * n
        mll_with_prior_i = model.eval_log_posterior_density(x_data, np.squeeze(y_data)).detach().numpy() * n
        mlls_gpytorch.append(mll_i)
        mlls_with_prior_gpytorch.append(mll_with_prior_i)
    assert np.allclose(np.array(mlls_gpytorch), mll, rtol=1e-3, atol=1e-2)
    assert np.allclose(np.array(mlls_with_prior_gpytorch), mll_with_prior, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("dim,active_dim", [(3, 1), (2, 0), (4, 2), (1, 0)])
def test_base_on_single_dims_kernels(dim, active_dim):
    X = np.random.rand(20, dim)
    X2 = np.random.rand(10, dim)
    X_torch = torch.from_numpy(X)
    X_torch2 = torch.from_numpy(X2)
    X_torch_sliced = torch.from_numpy(X)[:, active_dim].unsqueeze(-1)
    X_torch2_sliced = torch.from_numpy(X2)[:, active_dim].unsqueeze(-1)
    gpytorch_kernel_config = BasicRBFPytorchConfig(input_dimension=dim)
    gpytorch_kernel_config.active_on_single_dimension = True
    gpytorch_kernel_config.active_dimension = active_dim
    gpytorch_kernel_config.base_lengthscale = 0.7
    gpytorch_kernel_config.base_variance = 0.25
    se_kernel = SEKernel(4, [], 0.1, BaseKernelEvalMode.DEBUG)
    se_kernel.debug_lengthscale = torch.tensor(0.7)
    se_kernel.debug_variance = torch.tensor(0.5)
    K_torch = se_kernel.forward(X_torch_sliced, X_torch_sliced, None)
    K_gpytorch = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch).numpy()
    assert np.allclose(K_torch.numpy(), K_gpytorch)
    K_torch2 = se_kernel.forward(X_torch_sliced, X_torch2_sliced, None)
    K_gpytorch2 = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch2).numpy()
    assert np.allclose(K_torch2.numpy(), K_gpytorch2)

    gpytorch_kernel_config = BasicPeriodicKernelPytorchConfig(input_dimension=dim)
    gpytorch_kernel_config.active_on_single_dimension = True
    gpytorch_kernel_config.active_dimension = active_dim
    gpytorch_kernel_config.base_lengthscale = 0.7
    gpytorch_kernel_config.base_variance = 0.25
    gpytorch_kernel_config.base_period = 0.5
    per_kernel = PeriodicKernel(4, [], 0.1, BaseKernelEvalMode.DEBUG)
    per_kernel.debug_lengthscale = torch.tensor(0.7)
    per_kernel.debug_variance = torch.tensor(0.5)
    per_kernel.debug_period = torch.tensor(0.5)
    K_torch = per_kernel.forward(X_torch_sliced, X_torch_sliced, None)
    K_gpytorch = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch).numpy()
    assert np.allclose(K_torch.numpy(), K_gpytorch)
    K_torch2 = per_kernel.forward(X_torch_sliced, X_torch2_sliced, None)
    K_gpytorch2 = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch2).numpy()
    assert np.allclose(K_torch2.numpy(), K_gpytorch2)

    gpytorch_kernel_config = BasicLinearKernelPytorchConfig(input_dimension=dim)
    gpytorch_kernel_config.active_on_single_dimension = True
    gpytorch_kernel_config.active_dimension = active_dim
    lin_kernel = LinearKernel(4, [], 0.1, BaseKernelEvalMode.DEBUG)
    K_torch = lin_kernel.forward(X_torch_sliced, X_torch_sliced, None)
    K_gpytorch = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch).numpy()
    assert np.allclose(K_torch.numpy(), K_gpytorch)
    K_torch2 = lin_kernel.forward(X_torch_sliced, X_torch2_sliced, None)
    K_gpytorch2 = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch2).numpy()
    assert np.allclose(K_torch2.numpy(), K_gpytorch2)

    gpytorch_kernel_config = BasicMatern52PytorchConfig(input_dimension=dim)
    gpytorch_kernel_config.active_on_single_dimension = True
    gpytorch_kernel_config.active_dimension = active_dim
    gpytorch_kernel_config.base_lengthscale = 0.7
    gpytorch_kernel_config.base_variance = 0.25
    matern_kernel = MaternKernel(4, [], 0.1, 2.5, BaseKernelEvalMode.DEBUG)
    matern_kernel.debug_lengthscale = torch.tensor(0.7)
    matern_kernel.debug_variance = torch.tensor(0.5)
    K_torch = matern_kernel.forward(X_torch_sliced, X_torch_sliced, None)
    K_gpytorch = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch).numpy()
    assert np.allclose(K_torch.numpy(), K_gpytorch)
    K_torch2 = matern_kernel.forward(X_torch_sliced, X_torch2_sliced, None)
    K_gpytorch2 = PytorchKernelFactory.build(gpytorch_kernel_config)(X_torch, X_torch2).numpy()
    assert np.allclose(K_torch2.numpy(), K_gpytorch2)


def test_noise_predictor_shapes_and_masking():
    batch_size = 5
    X_list, y_list, X_padded, y_padded, N, size_mask, _, kernel_embeddings, kernel_mask, size_mask_kernel, kernel_list = get_test_inputs(
        batch_size, 10, 4, 1, only_SE=True
    )
    noise_predictor = NoiseVariancePredictorHead([10], N_BASE_KERNEL_TYPES, 1e-4, 0.2, False)
    num_base_kernels = 0
    for kernel_in_batch in kernel_list:
        for kernels_in_dim in kernel_in_batch:
            num_base_kernels += len(kernels_in_dim)
    num_base_kernels_2 = np.sum(noise_predictor.get_n_base_kernels(kernel_mask).numpy())
    assert np.allclose(num_base_kernels, num_base_kernels_2)


def test_batch_noise_marginal_likelhood():
    batch_size = 5
    X_list, y_list, X_padded, y_padded, N, size_mask, _, kernel_embeddings, _, size_mask_kernel, kernel_list = get_test_inputs(
        batch_size, 10, 4, 1, only_SE=True
    )
    kernel_wrapper = BatchedDimWiseAdditiveKernelWrapper(4, [], 0.1, True, BaseKernelEvalMode.DEBUG)
    K = kernel_wrapper.forward(X_padded, X_padded, kernel_embeddings, kernel_list)
    noise_variances = np.random.uniform(1e-4, 1.0, (batch_size, 1))
    mll = (
        cal_marg_likelihood_batch_noise(
            K, y_padded.unsqueeze(-1), torch.tensor(noise_variances).float(), size_mask_kernel, 1.0 - size_mask, N, torch.device("cpu")
        )[0]
        * N
    )
    mlls_gpytorch = []
    for i, x_data in enumerate(X_list):
        y_data = y_list[i]
        n = x_data.shape[0]
        d = x_data.shape[1]
        kernel_config = BasicRBFPytorchConfig(input_dimension=d)
        model_config = BasicGPModelPytorchConfig(kernel_config=kernel_config)
        model_config.optimize_hps = False
        model_config.initial_likelihood_noise = np.sqrt(np.squeeze(noise_variances[i]))
        model = ModelFactory.build(model_config)
        # model.kernel = gpflow.kernels.RBF()
        model.infer(x_data, np.expand_dims(y_data, axis=1))
        mll_i = model.eval_log_marginal_likelihood(x_data, np.squeeze(y_data)).detach().numpy() * n
        mlls_gpytorch.append(mll_i)
    assert np.allclose(np.array(mlls_gpytorch), mll)


def test_random_dataset_of_datasets():
    seed = 100
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    generator = GeneratorFactory.build(DimWiseAdditiveWithNoiseMixedNoMaternConfig())
    dataset = RandomDatasetOfDatasets(generator, True, 16, False, True)
    batch_size = 4
    dl = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=default_worker_init_function,
    )
    epochs = 2
    first_batch_per_epoch = []
    second_batch_per_epoch = []
    for _ in range(epochs):
        counter = 0
        random.seed(seed)
        np.random.seed(seed)
        for batch in dl:
            (_, _, X_padded, y_padded, _, _, _, _, _, _, _, _, _) = batch
            if counter == 0:
                first_batch_per_epoch.append(X_padded)
            elif counter == 1:
                second_batch_per_epoch.append(X_padded)
            counter += 1
            for i in range(batch_size - 1):
                for j in range(i + 1, batch_size):
                    assert not torch.equal(X_padded[j], X_padded[i])
                    assert not torch.equal(y_padded[j], y_padded[i])
    for i in range(0, epochs - 1):
        for j in range(i + 1, epochs):
            assert not torch.equal(first_batch_per_epoch[i], first_batch_per_epoch[j])
    for i in range(0, epochs):
        assert not torch.equal(first_batch_per_epoch[i], second_batch_per_epoch[i])


def test_value_list_kernel_wrapper():
    B = 10
    (
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
        kernel_list_of_lists,
    ) = get_test_inputs(B, 10, 4, 2, only_SE=False)
    kernel_wrapper = DimWiseAdditiveKernelWrapper(10, [], 0.1, True, BaseKernelEvalMode.VALUE_LIST)
    value_lists = []
    num_params_list = []
    for i, kernel_list in enumerate(kernel_list_of_lists):
        expression = transform_kernel_list_to_expression(kernel_list)
        state_dict = get_hp_sample_from_prior_gpytorch_as_state_dict(expression)
        kernel = get_gpytorch_kernel_from_expression_and_state_dict(expression, state_dict)
        value_list = get_parameter_value_lists(kernel_list, state_dict)
        X = X_list[i]
        X_torch = torch.from_numpy(X)
        K = kernel_wrapper.forward(X_torch, X_torch, kernel_embeddings[i], kernel_list, value_list)
        num_params = kernel_wrapper.get_num_params(kernel_list)
        num_params_list.append(num_params)
        value_lists.append(value_list)
        K_gpytorch = kernel(X_torch, X_torch).numpy()
        assert np.allclose(K, K_gpytorch, atol=1e-4, rtol=1e-4)
    batchified, lengths = get_batch_from_nested_parameter_list(value_lists)
    assert np.allclose(lengths, np.array(num_params_list))


if __name__ == "__main__":
    test_eval_amortized_model(DimWiseAdditiveKernelWithNoiseAmortizedModelConfig())
