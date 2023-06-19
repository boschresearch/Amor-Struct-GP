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
import argparse
import copy
from enum import Enum
from math import sqrt
import os
import random
import sys
from typing import List, Optional
from amorstructgp.config.kernels.gpytorch_kernels.elementary_kernels_pytorch_configs import BasicRBFPytorchConfig
from amorstructgp.config.models.gp_model_gpytorch_config import BasicGPModelPytorchConfig
from amorstructgp.config.prior_parameters import PRIOR_SETTING
from amorstructgp.data_generators.generator_factory import GeneratorFactory
from amorstructgp.config.data_generators.dim_wise_additive_generator_config import (
    BasicDimWiseAdditiveGeneratorConfig,
    DimWiseAdditiveWithNoiseMixedNoMaternConfig,
    DimWiseAdditiveWithNoiseMixedNoMaternNoNegativeConfig,
    DimWiseAdditiveWithNoiseMixedNoMaternBiggerConfig,
    DimWiseAdditiveWithNoiseMixedNoMaternVeryLargeConfig,
    DimWiseAdditiveWithNoiseMixedNoMaternMoreGTConfig,
    DimWiseAdditiveWithNoiseMixedNoMaternBiggerOnlyPositiveConfig,
    DimWiseAdditiveWithNoiseMixedWithMaternBiggerOnlyPositiveConfig,
    DimWiseAdditiveWithNoiseMixedWithMaternBiggerConfig,
    OneDTimeSeriesAdditiveWithNoiseMixedNoMaternBiggerConfig,
    OneDTimeSeriesAdditiveWithNoiseOnlyPositiveNoMaternBiggerConfig,
)
from amorstructgp.config.nn.amortized_infer_models_configs import (
    BasicDimWiseAdditiveKernelAmortizedModelConfig,
    CrossAttentionKernelEncSharedDatasetEncAmortizedModelConfig,
    DimWiseAdditiveKernelWithNoiseAmortizedModelConfig,
    ExperimentalAmortizedModelConfig,
    ExperimentalAmortizedModelConfig2,
    SmallerDimWiseAdditiveKernelWithNoiseAmortizedModelConfig,
    SharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig,
    SmallerSharedDatasetEncodingDimWiseAdditiveAmortizedModelConfig,
    FullDimGlobalNoiseDimWiseAdditiveAmortizedModelConfig,
    BiggerCrossAttentionKernelEncSharedDatasetEncAmortizedModelConfig,
    WiderCrossAttentionKernelEncSharedDatasetEncAmortizedModelConfig,
    WiderCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    WiderStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    DeeperCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    SmallerStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    ExperimentalCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    ExperimentalSmallerStandardCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    SmallerStandardSmallNoiseBoundNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig,
    ARDRBFOnlyAmortizedModelConfig,
    BiggerARDRBFOnlyAmortizedModelConfig,
)
from amorstructgp.config.training.training_configs import BasicDimWiseAdditiveKernelTrainingConfig,DimWiseAdditiveKernelFineTuningConfig
from amorstructgp.data_generators.dataset_of_datasets import (
    DatasetOfDatasets,
    DatasetType,
    RandomDatasetOfDatasets,
)
from amorstructgp.data_generators.dim_wise_additive_kernels_generator import (
    DimWiseAdditiveKernelGenerator,
)
from amorstructgp.data_generators.simulator import SimulatedDataset
from amorstructgp.gp.base_kernels import (
    get_batch_from_nested_parameter_list,
    transform_kernel_list_to_expression,
)
from amorstructgp.nn.amortized_inference_models import (
    BasicDimWiseAdditiveAmortizedInferenceModel,
    DimWiseAdditiveKernelsAmortizedInferenceModel,
)
from amorstructgp.nn.amortized_models_factory import AmortizedModelsFactory
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import logging
import numpy as np
from amorstructgp.utils.enums import LossFunctionType, OptimizerType
from amorstructgp.utils.utils import default_worker_init_function, plot_predictions
from amorstructgp.models.gp_model_pytorch import GPModelPytorch
from amorstructgp.utils.utils import (
    calculate_nll_normal,
    calculate_rmse,
    get_datetime_as_string,
    write_dict_to_json,
)
from amorstructgp.utils.gpytorch_utils import get_gpytorch_kernel_from_expression_and_state_dict
from copy import deepcopy
import pyro
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CheckpointStrategy(str,Enum):
    RESUME_COMPLETE = "RESUME_COMPLETE"
    RESUME_MODEL_AND_OPT = "RESUME_MODEL_AND_OPT"
    RESUME_MODEL = "RESUME_MODEL"


class Trainer:
    def __init__(
        self,
        model_config: BasicDimWiseAdditiveKernelAmortizedModelConfig,
        dataset_folder: str,
        output_folder: str,
        training_config: BasicDimWiseAdditiveKernelTrainingConfig,
        run_suffix: str = "",
        generate_dataset_on_the_fly: bool = False,
        generator_config: Optional[BasicDimWiseAdditiveGeneratorConfig] = None,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.generator_config = generator_config
        self.generate_dataset_on_the_fly = generate_dataset_on_the_fly
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.fixed_likelihood_variance = model_config.has_fix_noise_variance
        # Create run output folder
        self.run_string = "run_{}{}".format(get_datetime_as_string(), run_suffix)
        self.run_folder = os.path.join(self.output_folder, self.run_string)
        self.create_folder(self.run_folder)

        # Initialize tensorboard summary writer object
        self.tensorboard_writer = SummaryWriter(self.run_folder)

        # Save configs and write to tensorboard
        self.save_config(
            training_config.dict(),
            model_config.dict(),
            self.dataset_folder,
            self.run_folder,
            (None if not generate_dataset_on_the_fly else generator_config.dict()),
        )
        self.write_configs_to_tensorboard(model_config, training_config, generate_dataset_on_the_fly, generator_config)

        # Create snapshot folder
        self.snapshot_folder = os.path.join(self.run_folder, "snapshots")
        self.create_folder(self.snapshot_folder)
        self.do_snapshots = True
        self.snapshot_interval = 100

        self.loss_function_type = training_config.loss_function_type
        self.num_epochs = training_config.num_epochs
        self.batch_size = training_config.batch_size
        self.learning_rate = training_config.learning_rate
        self.normalize_datasets = training_config.normalize_datasets
        self.warp_inputs = training_config.warp_inputs
        self.use_lr_scheduler = training_config.use_lr_scheduler
        self.optimizer_type = training_config.optimizer_type
        self.freeze_dataset_encoder = training_config.freeze_dataset_encoder

        if (
            self.loss_function_type == LossFunctionType.PARAMETER_RMSE
            or self.loss_function_type == LossFunctionType.PARAMETER_RMSE_PLUS_NMLL
        ):
            self.return_kernel_parameter_lists = True
        else:
            self.return_kernel_parameter_lists = False

        # Initialize dataset objects - only datapoint indexes/uuids are loaded - no actual data is loaded yet
        if self.generate_dataset_on_the_fly:
            # In this case data is sampled for each batch on the fly - no loading from a fixed dataset form files
            # This also means that in each epoch new batches are drawn
            generator = GeneratorFactory.build(self.generator_config)
            self.train_dataset = RandomDatasetOfDatasets(
                generator,
                self.normalize_datasets,
                self.generator_config.num_train_datasets_on_the_fly,
                self.warp_inputs,
                self.return_kernel_parameter_lists,
            )
            self.val_dataset = RandomDatasetOfDatasets(
                generator,
                self.normalize_datasets,
                self.generator_config.num_val_datasets_on_the_fly,
                self.warp_inputs,
                self.return_kernel_parameter_lists,
            )
            self.test_dataset = RandomDatasetOfDatasets(
                generator,
                self.normalize_datasets,
                self.generator_config.num_test_datasets_on_the_fly,
                self.warp_inputs,
                self.return_kernel_parameter_lists,
            )
        else:
            self.train_dataset = DatasetOfDatasets(
                DimWiseAdditiveKernelGenerator,
                dataset_folder,
                DatasetType.TRAIN,
                self.normalize_datasets,
                self.warp_inputs,
                self.return_kernel_parameter_lists,
            )
            self.val_dataset = DatasetOfDatasets(
                DimWiseAdditiveKernelGenerator,
                dataset_folder,
                DatasetType.VAL,
                self.normalize_datasets,
                self.warp_inputs,
                self.return_kernel_parameter_lists,
            )
            self.test_dataset = DatasetOfDatasets(
                DimWiseAdditiveKernelGenerator,
                dataset_folder,
                DatasetType.TEST,
                self.normalize_datasets,
                self.warp_inputs,
                self.return_kernel_parameter_lists,
            )

        self.val_epochs = 20
        self.n_eval_datasets_per_dim = 50
        self.max_dim_eval_datasets = 4
        self.num_workers = 5
        self.initial_seed = 100

        # Initialize gpytorch model for validation and plotting
        self.gpytorch_model_config = BasicGPModelPytorchConfig(kernel_config=BasicRBFPytorchConfig(input_dimension=0))
        self.gpytorch_model_config.add_constant_mean_function = False
        if self.fixed_likelihood_variance:
            self.gpytorch_model_config.initial_likelihood_noise = sqrt(self.model_config.gp_variance)
            self.gpytorch_model_config.fix_likelihood_variance = True
        else:
            self.gpytorch_model_config.fix_likelihood_variance = False

        # Always keep state dict of model with best validation error - save it in the end
        self.best_model_dict = {}
        self.best_model_dict["state_dict"] = None
        self.best_model_dict["validation_error"] = np.inf
        self.best_model_dict["epoch_index"] = None

        self.checkpoint_strategy = CheckpointStrategy.RESUME_COMPLETE

    def write_configs_to_tensorboard(self, model_config, training_config, generate_dataset_on_the_fly, generator_config):
        self.tensorboard_writer.add_text("model_config", str(model_config.dict()))
        self.tensorboard_writer.add_text("training_config", str(training_config.dict()))
        if generate_dataset_on_the_fly:
            self.tensorboard_writer.add_text("generator_config", str(generator_config.dict()))
        else:
            self.tensorboard_writer.add_text("dataset_folder", self.dataset_folder)

    def train(self, use_gpu: bool, use_checkpoint: bool = False, checkpoint_file_path: str = None):
        self.set_initial_seed(self.initial_seed)
        device = torch.device("cuda" if use_gpu else "cpu")
        epoch_start_index = 0

        ################# Produce Evaluation datasets  #######################

        # first step - to ensure that for the same seed always the same datasets are produced

        evaluate_on_train_datasets = not self.generate_dataset_on_the_fly

        # create datasets for evaluation + train standard gp model on them and return learned parameters
        (
            evaluation_datasets_dict_train,
            evaluation_datasets_dict_val,
            max_dimension_eval,
            evaluation_datasets_ml_param_train,
            evaluation_datasets_ml_param_val,
        ) = self.get_eval_datasets_and_ml_params(evaluate_on_train_datasets)

        ################## Build main model ####################################

        model = AmortizedModelsFactory.build(self.model_config)

        if use_gpu:
            model = model.to(device)
        assert isinstance(model, BasicDimWiseAdditiveAmortizedInferenceModel)

        if self.freeze_dataset_encoder:
            model.freeze_dataset_encoder()

        train_loader, val_loader, test_loader = self.initialize_data_loader()

        ##################### Initialize optimizer #############################
        optimizer, scheduler = self.initialize_optimizer_and_scheduler(model, train_loader)
        self.global_step = 0

        ##################### Load model and optimizer from checkpoint ########################
        if use_checkpoint:
            epoch_start_index = self.load_states_from_checkpoint(checkpoint_file_path, model, optimizer, scheduler)

        optimizer.zero_grad()

        for epoch_index in range(epoch_start_index, self.num_epochs):
            ################## Validation #########################
            if epoch_index % self.val_epochs == 0:
                model.eval()

                if (
                    evaluate_on_train_datasets
                ):  # it does not make sense to distinguish between train and val set if train set is resampled in each epoch anyway and the distributions are the same
                    # calculate complete loss on train set
                    # add 1D train prediction plots
                    self.add_prediction_plots(
                        device, model, evaluation_datasets_dict_train[1], evaluation_datasets_ml_param_train[1], epoch_index, "train"
                    )
                    # add 2D train prediction plots
                    if 2 in evaluation_datasets_dict_train:
                        self.add_prediction_plots(
                            device, model, evaluation_datasets_dict_train[2], evaluation_datasets_ml_param_train[2], epoch_index, "train"
                        )
                    train_mean_nmll_complete, _, train_global_loss = self.calculate_full_dataset_loss(
                        model, train_loader, epoch_index, device
                    )
                    self.tensorboard_writer.add_scalar("train_nmll_complete", train_mean_nmll_complete, global_step=epoch_index)
                    self.tensorboard_writer.add_scalar("train_loss_complete", train_global_loss, global_step=epoch_index)

                    self.evaluate_prediction_on_dataset_samples(
                        device,
                        model,
                        evaluation_datasets_dict_train,
                        max_dimension_eval,
                        evaluation_datasets_ml_param_train,
                        epoch_index,
                        "train",
                    )

                # add 1D val prediction plots
                self.add_prediction_plots(
                    device, model, evaluation_datasets_dict_val[1], evaluation_datasets_ml_param_val[1], epoch_index, "val"
                )

                # add 2D val prediction plots
                if 2 in evaluation_datasets_dict_val:
                    self.add_prediction_plots(
                        device, model, evaluation_datasets_dict_val[2], evaluation_datasets_ml_param_val[2], epoch_index, "val"
                    )

                # calculate complete loss on val set
                val_mean_nmll_complete, _, val_global_loss = self.calculate_full_dataset_loss(model, val_loader, epoch_index, device)
                self.tensorboard_writer.add_scalar("val_nmll_complete", val_mean_nmll_complete, global_step=epoch_index)
                self.tensorboard_writer.add_scalar("val_loss_complete", val_global_loss, global_step=epoch_index)

                # Evaluation on sample datasets - to compare predictive performance against gt and ml
                self.evaluate_prediction_on_dataset_samples(
                    device, model, evaluation_datasets_dict_val, max_dimension_eval, evaluation_datasets_ml_param_val, epoch_index, "val"
                )

                if val_global_loss < self.best_model_dict["validation_error"]:
                    model.to("cpu")
                    self.best_model_dict["validation_error"] = val_global_loss
                    self.best_model_dict["state_dict"] = copy.deepcopy(model.state_dict())
                    self.best_model_dict["epoch_index"] = epoch_index
                    model.to(device)

            # Save model
            if self.do_snapshots and epoch_index % self.snapshot_interval == 0 and epoch_index > 0:
                # self.save_model(model.state_dict(), epoch_index, self.snapshot_folder)
                self.save_checkpoint(
                    model.state_dict(),
                    optimizer.state_dict(),
                    (None if not self.use_lr_scheduler else scheduler.state_dict()),
                    epoch_index,
                    self.global_step,
                    self.snapshot_folder,
                )

            ################## Training  ##########################
            print("Next epoch")
            model.train()
            for batch_index, batch in enumerate(train_loader):
                if use_gpu:
                    batch = self.move_to_device(batch, device)
                (
                    kernel_embeddings,
                    kernel_mask,
                    X_padded,
                    y_padded,
                    size_mask,
                    dim_mask,
                    size_mask_kernel,
                    N,
                    kernel_list,
                    mlls_gt,
                    log_posterior_density_gt,
                    gt_kernel_parameter_lists,
                    observation_noise_gt,
                ) = batch
                # self.print_shapes(X_padded, y_padded, N, size_mask, dim_mask, kernel_embeddings, kernel_mask, size_mask_kernel)
                optimizer.zero_grad()
                print("model forward")
                # Forward pass
                # kernel_embeddings, K, nmll, noise_variances,mll_success
                (
                    kernel_embeddings_out,
                    K_out,
                    nmlls_over_batch,
                    nmlls_with_prior_over_batch,
                    noise_variances_out,
                    log_prior_prob_kernel_params,
                    log_prior_prob_variance,
                    nmll_success,
                ) = model.forward(
                    X_padded, y_padded, N, size_mask, dim_mask, kernel_embeddings, kernel_mask, size_mask_kernel, kernel_list, device=device
                )
                num_predicted_params = model.get_num_predicted_params(kernel_list, device=device)
                if (
                    self.loss_function_type == LossFunctionType.PARAMETER_RMSE
                    or self.loss_function_type == LossFunctionType.PARAMETER_RMSE_PLUS_NMLL
                ):
                    pred_kernel_parameter_list = model.get_parameter_nested_lists(kernel_embeddings_out, kernel_list, detach=False)
                    batch_predicted_params = get_batch_from_nested_parameter_list(pred_kernel_parameter_list, device)[0]
                    batch_gt_params = get_batch_from_nested_parameter_list(gt_kernel_parameter_lists)[0].to(device)
                else:
                    pred_kernel_parameter_list = None
                    batch_predicted_params = None
                    batch_gt_params = None
                if nmll_success:
                    # Calculate loss
                    mean_nmlls_train = torch.mean(nmlls_over_batch)
                    train_loss = self.loss_function(
                        nmlls_over_batch,
                        nmlls_with_prior_over_batch,
                        log_prior_prob_kernel_params,
                        log_prior_prob_variance,
                        N,
                        num_predicted_params,
                        batch_predicted_params,
                        batch_gt_params,
                        noise_variances_out,
                        observation_noise_gt,
                    )

                    # Make gradient step
                    train_loss.backward()
                    optimizer.step()

                    if self.use_lr_scheduler:
                        # Schedule Learning rate
                        last_lr = scheduler.get_last_lr()[0]
                        # last_lr = last_lr.detach().cpu().numpy()
                        scheduler.step()
                    else:
                        last_lr = self.learning_rate

                    # Tensorboard writting and metrics logging
                    mean_nmlls_gt = -1.0 * torch.mean(mlls_gt)

                    mean_nmlls_train = float(mean_nmlls_train.detach().cpu().numpy())
                    mean_nmlls_gt = float(mean_nmlls_gt.detach().cpu().numpy())
                    mean_nmlls_diff = mean_nmlls_train - mean_nmlls_gt

                    self.tensorboard_writer.add_scalar("train_loss", train_loss, global_step=self.global_step)
                    self.tensorboard_writer.add_scalar("mean_nmlls_train", mean_nmlls_train, global_step=self.global_step)
                    self.tensorboard_writer.add_scalar("mean_nmlls_gt", mean_nmlls_gt, global_step=self.global_step)
                    self.tensorboard_writer.add_scalar("mean_nmlls_diff", mean_nmlls_diff, global_step=self.global_step)
                    self.tensorboard_writer.add_scalar("learning_rate", last_lr, global_step=self.global_step)

                    print(
                        "Epoch: {} - Batch: {} - Train-Loss: {} - Avg-NMLL-GT: {}".format(
                            epoch_index, batch_index, mean_nmlls_train, mean_nmlls_gt
                        )
                    )

                    self.global_step += train_loader.batch_size
                else:
                    # We propagate chol error that occur inside torch.linalg.cholesky_ex for a batch to here and delete all tensors that occur in the forward pass to clean up GPU memory
                    print("Skip Epoch: {} - Batch: {} because of chol error".format(epoch_index, batch_index))
                    del (
                        nmlls_over_batch,
                        noise_variances_out,
                        nmlls_with_prior_over_batch,
                        log_prior_prob_kernel_params,
                        log_prior_prob_variance,
                        K_out,
                        kernel_embeddings_out,
                        pred_kernel_parameter_list,
                        batch_predicted_params,
                        num_predicted_params,
                        batch_gt_params,
                    )
                    del (
                        kernel_embeddings,
                        kernel_mask,
                        X_padded,
                        y_padded,
                        size_mask,
                        dim_mask,
                        size_mask_kernel,
                        N,
                        kernel_list,
                        mlls_gt,
                        log_posterior_density_gt,
                        gt_kernel_parameter_lists,
                        observation_noise_gt,
                    )
                    del batch
                    torch.cuda.empty_cache()

            if use_gpu and epoch_index == epoch_start_index + 1:
                self.show_cuda_memory_info()

        # Save best model
        best_model_state_dict = self.best_model_dict["state_dict"]
        epoch_best_model = self.best_model_dict["epoch_index"]
        print("Best model validation error: {}".format(self.best_model_dict["validation_error"]))
        self.save_model(best_model_state_dict, epoch_best_model, self.run_folder)

    def get_eval_datasets_and_ml_params(self, evaluate_on_train_datasets):
        # A subset of the train and val datasets are sampled - ML inference is performed on them for comparision

        # Dicts that store lists of SimulatedDataset objects for each key=dimenion
        evaluation_datasets_dict_train = {}
        evaluation_datasets_dict_val = {}
        max_dimension_train = self.train_dataset.get_max_dimension()
        max_dimension_val = self.val_dataset.get_max_dimension()

        # Dicts that store a Tuple of Lists containing the learned kernel parameters and noise variance for each Simulated Dataset object (for each key=dimension)
        evaluation_datasets_ml_param_train = {}
        evaluation_datasets_ml_param_val = {}
        assert max_dimension_train == max_dimension_val
        max_dimension = min(max_dimension_train, self.max_dim_eval_datasets)
        for d in range(1, max_dimension + 1):
            # Produce sample from the dataset of datasets (each with input_dimension=d)
            val_datasets_sample_for_dim = self.val_dataset.get_n_datasets_with_input_dimension(self.n_eval_datasets_per_dim, d)

            # Check some consistencies (fail early if possible)
            assert len(val_datasets_sample_for_dim) == self.n_eval_datasets_per_dim
            assert val_datasets_sample_for_dim[0].prior_setting == PRIOR_SETTING
            if self.fixed_likelihood_variance:
                assert np.allclose(val_datasets_sample_for_dim[0].observation_noise, np.sqrt(self.model_config.gp_variance))
                assert np.allclose(val_datasets_sample_for_dim[0].observation_noise, self.gpytorch_model_config.initial_likelihood_noise)
            val_datasets_sample_ml_parameters = self.standard_GP_learning(val_datasets_sample_for_dim, add_prior=False)
            evaluation_datasets_dict_val[d] = val_datasets_sample_for_dim
            evaluation_datasets_ml_param_val[d] = val_datasets_sample_ml_parameters

            if evaluate_on_train_datasets:
                # Perform ML inference for each element in the list of datasets
                train_datasets_sample_for_dim = self.train_dataset.get_n_datasets_with_input_dimension(self.n_eval_datasets_per_dim, d)
                assert len(train_datasets_sample_for_dim) == self.n_eval_datasets_per_dim
                train_datasets_sample_ml_parameters = self.standard_GP_learning(train_datasets_sample_for_dim, add_prior=False)
                evaluation_datasets_dict_train[d] = train_datasets_sample_for_dim
                evaluation_datasets_ml_param_train[d] = train_datasets_sample_ml_parameters
        return (
            evaluation_datasets_dict_train,
            evaluation_datasets_dict_val,
            max_dimension,
            evaluation_datasets_ml_param_train,
            evaluation_datasets_ml_param_val,
        )

    def initialize_optimizer_and_scheduler(self, model, train_loader):
        # Initialize optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())

        if self.optimizer_type == OptimizerType.ADAM:
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_type == OptimizerType.RADAM:
            optimizer = torch.optim.RAdam(params, lr=self.learning_rate)
        else:
            raise ValueError

        # Initialize learching rate scheduler
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.learning_rate, steps_per_epoch=len(train_loader), epochs=self.num_epochs
            )
        else:
            scheduler = None
        return optimizer, scheduler

    def load_states_from_checkpoint(self, checkpoint_file_path, model, optimizer, scheduler):
        (
            model_state_dict,
            optimizer_state_dict,
            scheduler_state_dict,
            checkpoint_epoch_index,
            checkpoint_global_step,
            model_config_dict,
            training_config_dict,
        ) = self.load_checkpoint(checkpoint_file_path)
        # assert self.model_config.dict() == model_config_dict
        if self.checkpoint_strategy == CheckpointStrategy.RESUME_MODEL:
            model.load_state_dict(model_state_dict)
            epoch_start_index = 0
        elif self.checkpoint_strategy == CheckpointStrategy.RESUME_MODEL_AND_OPT:
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            epoch_start_index = checkpoint_epoch_index
            self.global_step = checkpoint_global_step
        elif self.checkpoint_strategy == CheckpointStrategy.RESUME_COMPLETE:
            assert self.training_config.dict() == training_config_dict
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            if self.training_config.use_lr_scheduler:
                scheduler.load_state_dict(scheduler_state_dict)
            epoch_start_index = checkpoint_epoch_index
            self.global_step = checkpoint_global_step
        else:
            raise ValueError()
        return epoch_start_index

    def initialize_data_loader(self):
        # Initialize data loader that iterate through the datasets via batches (worker_init_fn makes sure that np seeds are different over batches and epochs)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=default_worker_init_function,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=default_worker_init_function,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=default_worker_init_function,
        )

        return train_loader, val_loader, test_loader

    def loss_function(
        self,
        nmll: torch.tensor,
        nmll_with_prior: torch.tensor,
        log_prior_prob_kernel_params: torch.tensor,
        log_prior_prob_variance: torch.tensor,
        num_data: torch.tensor,
        num_parameters: torch.tensor,
        batch_predicted_params: torch.tensor,
        batch_gt_params: torch.tensor,
        noise_variances_out: torch.tensor,
        observation_noise_gt: torch.tensor,
    ):
        if self.loss_function_type == LossFunctionType.NMLL:
            loss_function_value = torch.mean(nmll)
        elif self.loss_function_type == LossFunctionType.NMLL_WITH_PRIOR:
            loss_function_value = torch.mean(nmll_with_prior)
        elif self.loss_function_type == LossFunctionType.NMLL_PARAM_SCALED:
            scaled_nmll = nmll / num_parameters
            loss_function_value = torch.mean(scaled_nmll)
        elif self.loss_function_type == LossFunctionType.NMLL_WITH_PRIOR_PARAM_SCALED:
            scaled_nmll_with_prior = nmll_with_prior / num_parameters
            loss_function_value = torch.mean(scaled_nmll_with_prior)
        elif self.loss_function_type == LossFunctionType.NMLL_SQRT_PARAM_SCALED:
            scaled_nmll = nmll / torch.sqrt(num_parameters)
            loss_function_value = torch.mean(scaled_nmll)
        elif self.loss_function_type == LossFunctionType.NMLL_WITH_PRIOR_SQRT_PARAM_SCALED:
            scaled_nmll_with_prior = nmll_with_prior / torch.sqrt(num_parameters)
            loss_function_value = torch.mean(scaled_nmll_with_prior)
        elif self.loss_function_type == LossFunctionType.NMLL_WITH_NOISE_PRIOR:
            nmll_with_noise_prior = nmll - log_prior_prob_variance / num_data
            loss_function_value = torch.mean(nmll_with_noise_prior)
        elif self.loss_function_type == LossFunctionType.PARAMETER_RMSE:
            observation_noise_out = torch.sqrt(noise_variances_out)
            batch_size = batch_predicted_params.shape[0]
            n_kernel_params = num_parameters - 1
            mse_kernel_params = (
                torch.sum(torch.square(batch_predicted_params - batch_gt_params) / n_kernel_params.unsqueeze(-1)) / batch_size
            )
            mse_observation_noise = torch.mean(torch.square(observation_noise_out - observation_noise_gt))
            loss_function_value = mse_kernel_params + mse_observation_noise
        elif self.loss_function_type == LossFunctionType.PARAMETER_RMSE_PLUS_NMLL:
            observation_noise_out = torch.sqrt(noise_variances_out)
            batch_size = batch_predicted_params.shape[0]
            n_kernel_params = num_parameters - 1
            mse_kernel_params = (
                torch.sum(torch.square(batch_predicted_params - batch_gt_params) / n_kernel_params.unsqueeze(-1)) / batch_size
            )
            # @TODO: check weighting here
            mse_observation_noise = torch.mean(torch.square(observation_noise_out - observation_noise_gt))
            loss_function_value1 = mse_kernel_params + mse_observation_noise
            loss_function_value2 = torch.mean(nmll)
            loss_function_value = 10.0 * loss_function_value1 + loss_function_value2
        elif self.loss_function_type == LossFunctionType.NOISE_RMSE_PLUS_NMLL:
            observation_noise_out = torch.sqrt(noise_variances_out)
            mse_observation_noise = torch.mean(torch.square(observation_noise_out - observation_noise_gt))
            loss_function_value1 = mse_observation_noise
            loss_function_value2 = torch.mean(nmll)
            loss_function_value = 10.0 * loss_function_value1 + loss_function_value2

        return loss_function_value

    def evaluate_prediction_on_dataset_samples(
        self, device, model, evaluation_datasets_dict, max_dimension, evaluation_datasets_ml_param, epoch_index, suffix
    ):
        for d in range(1, max_dimension + 1):
            rmses_ml_val, rmses_amortized_val, nlls_ml_val, nlls_amortized_val = self.eval_on_dataset_sample(
                device, model, evaluation_datasets_dict[d], evaluation_datasets_ml_param[d]
            )
            self.tensorboard_writer.add_scalar(
                "RMSE_diff_ml_{}/dim_{}".format(suffix, d), np.mean(rmses_amortized_val) - np.mean(rmses_ml_val), global_step=epoch_index
            )

            self.tensorboard_writer.add_scalar(
                "NLL_diff_ml_{}/dim_{}".format(suffix, d), np.mean(nlls_amortized_val) - np.mean(nlls_ml_val), global_step=epoch_index
            )

    def calculate_full_dataset_loss(self, model, data_loader, epoch_index, device):
        """
        Calcuate loss over complete dataset - only for validation - returns numpy array
        """
        assert isinstance(data_loader.dataset, DatasetOfDatasets) or isinstance(data_loader.dataset, RandomDatasetOfDatasets)
        with torch.no_grad():
            sum_over_nmlls = torch.tensor(0.0).to(device)
            sum_over_nmlls_gt = torch.tensor(0.0)
            global_loss = torch.tensor(0.0).to(device)
            N_dataset = len(data_loader.dataset)
            for batch_index, batch in enumerate(data_loader):
                # Move batch to respective device
                batch = self.move_to_device(batch, device)
                (
                    kernel_embeddings,
                    kernel_mask,
                    X_padded,
                    y_padded,
                    size_mask,
                    dim_mask,
                    size_mask_kernel,
                    N,
                    kernel_list,
                    mlls_gt,
                    log_posterior_density_gt,
                    gt_kernel_parameter_lists,
                    observation_noise_gt,
                ) = batch
                batch_size = len(kernel_list)
                # Get NMLLs over batch via forward pass
                (
                    kernel_embeddings_out,
                    K_out,
                    nmlls_over_batch,
                    nmlls_with_prior_over_batch,
                    noise_variances_out,
                    log_prior_prob_kernel_params,
                    log_prior_prob_variance,
                    nmll_success,
                ) = model.forward(
                    X_padded, y_padded, N, size_mask, dim_mask, kernel_embeddings, kernel_mask, size_mask_kernel, kernel_list, device=device
                )
                num_predicted_params = model.get_num_predicted_params(kernel_list, device=device)
                if (
                    self.loss_function_type == LossFunctionType.PARAMETER_RMSE
                    or self.loss_function_type == LossFunctionType.PARAMETER_RMSE_PLUS_NMLL
                ):
                    pred_kernel_parameter_list = model.get_parameter_nested_lists(kernel_embeddings_out, kernel_list, detach=False)
                    batch_predicted_params = get_batch_from_nested_parameter_list(pred_kernel_parameter_list, device)[0]
                    batch_gt_params = get_batch_from_nested_parameter_list(gt_kernel_parameter_lists)[0].to(device)
                else:
                    batch_predicted_params = None
                    batch_gt_params = None
                if nmll_success:
                    global_loss += (
                        self.loss_function(
                            nmlls_over_batch,
                            nmlls_with_prior_over_batch,
                            log_prior_prob_kernel_params,
                            log_prior_prob_variance,
                            N,
                            num_predicted_params,
                            batch_predicted_params,
                            batch_gt_params,
                            noise_variances_out,
                            observation_noise_gt,
                        )
                        * batch_size
                    )
                    sum_over_nmlls += torch.sum(nmlls_over_batch)
                    sum_over_nmlls_gt += -1.0 * torch.sum(mlls_gt)
                else:
                    N_dataset = N_dataset - batch_size
                    del (
                        nmlls_over_batch,
                        K_out,
                        nmlls_with_prior_over_batch,
                        log_prior_prob_kernel_params,
                        log_prior_prob_variance,
                        num_predicted_params,
                        pred_kernel_parameter_list,
                        noise_variances_out,
                        batch_predicted_params,
                        batch_gt_params,
                        kernel_embeddings_out,
                    )
                    del (
                        kernel_embeddings,
                        kernel_mask,
                        X_padded,
                        y_padded,
                        size_mask,
                        dim_mask,
                        size_mask_kernel,
                        N,
                        kernel_list,
                        mlls_gt,
                        log_posterior_density_gt,
                        gt_kernel_parameter_lists,
                        observation_noise_gt,
                    )
                    del batch
                    torch.cuda.empty_cache()
            mean_nmll_dataset = sum_over_nmlls / N_dataset
            mean_nmll_dataset_gt = sum_over_nmlls_gt / N_dataset
            global_loss = global_loss / N_dataset
            mean_nmll_dataset = torch.squeeze(mean_nmll_dataset).cpu().numpy()
            mean_nmll_dataset_gt = torch.squeeze(mean_nmll_dataset_gt).cpu().numpy()
            global_loss = torch.squeeze(global_loss).cpu().numpy()
        return mean_nmll_dataset, mean_nmll_dataset_gt, global_loss

    def set_initial_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def add_prediction_plots(self, device, model, simulated_datasets_sample, ml_learned_parameters_sample, epoch_index, prefix):
        """
        PLot predictions of the amortized model against gt model (@TODO: against standard GP with ML type 2) on one
        dimensional datasets
        """
        learned_kernel_parameters, learned_noise_variances = ml_learned_parameters_sample
        for i, simulated_dataset in enumerate(simulated_datasets_sample):
            # try:
            assert isinstance(simulated_dataset, SimulatedDataset)
            kernel_expression_gt = simulated_dataset.get_kernel_expression_gt()
            kernel_list_gt = simulated_dataset.get_kernel_list_gt()
            input_kernel_list = simulated_dataset.get_input_kernel_list()
            kernel_state_dict = simulated_dataset.get_kernel_state_dict_gt()
            observation_noise = simulated_dataset.get_observation_noise()
            ml_learned_observation_noise = np.sqrt(learned_noise_variances[i])
            ml_learned_kernel_state_dict = learned_kernel_parameters[i]
            if self.fixed_likelihood_variance:
                assert np.allclose(observation_noise, np.sqrt(self.model_config.gp_variance))
                assert np.allclose(observation_noise, ml_learned_observation_noise)
                assert np.allclose(observation_noise, self.gpytorch_model_config.initial_likelihood_noise)
            # get ground truth kernel (strucutre + parameters) with which dataset was generated
            kernel_gt = get_gpytorch_kernel_from_expression_and_state_dict(kernel_expression_gt, kernel_state_dict, wrap_in_addition=True)
            kernel_expression_ml = transform_kernel_list_to_expression(input_kernel_list, add_prior=False)
            kernel_ml = get_gpytorch_kernel_from_expression_and_state_dict(
                kernel_expression_ml, ml_learned_kernel_state_dict, wrap_in_addition=False
            )
            self.gpytorch_model_config.optimize_hps = False
            gt_gyptroch_model_config = deepcopy(self.gpytorch_model_config)
            gt_gyptroch_model_config.initial_likelihood_noise = observation_noise
            ml_gyptroch_model_config = deepcopy(self.gpytorch_model_config)
            ml_gyptroch_model_config.initial_likelihood_noise = ml_learned_observation_noise
            # Initialize GP model with gt kernel - set hp training to false
            gpytorch_model_gt = GPModelPytorch(kernel=kernel_gt, **gt_gyptroch_model_config.dict())
            gpytorch_model_ml = GPModelPytorch(kernel=kernel_ml, **ml_gyptroch_model_config.dict())
            X_data, y_data = simulated_dataset.get_dataset(self.normalize_datasets)
            X_test, y_test = simulated_dataset.get_test_dataset(self.normalize_datasets)
            # use different y_data for gt as normalization of dataset needs to ignored in this case
            y_data_for_gt = simulated_dataset.y_data
            y_test_for_gt = simulated_dataset.y_test
            gpytorch_model_gt.infer(X_data, y_data_for_gt)
            gpytorch_model_ml.infer(X_data, y_data)
            input_dimension = X_data.shape[1]
            if input_dimension == 1:
                X_eval = np.expand_dims(np.linspace(-0.5, 1.5, 200), axis=1)
            elif input_dimension == 2:
                X_eval = np.random.uniform(0, 1, size=(400, 2))
            else:
                raise ValueError
            # predict with both models
            pred_mu_gpytorch_gt, pred_sigma_gpytorch_gt = gpytorch_model_gt.predictive_dist(X_eval)
            pred_mu_gpytorch_ml, pred_sigma_gpytorch_ml = gpytorch_model_ml.predictive_dist(X_eval)
            pred_mu_model, _, pred_sigma_model = model.predict(X_eval, X_data, y_data, input_kernel_list, device=device)
            kernel_expression_gt_description = str(kernel_expression_gt)
            kernel_expression_ml_description = str(kernel_expression_ml)
            if input_kernel_list == kernel_list_gt:
                desciption = kernel_expression_gt_description
            else:
                desciption = kernel_expression_ml_description + " Ground-Truth: " + kernel_expression_gt_description
            figure = plot_predictions(
                X_data,
                y_data,
                y_data_for_gt,
                y_data,
                X_test,
                y_test,
                y_test_for_gt,
                y_test,
                X_eval,
                pred_mu_model,
                pred_sigma_model,
                pred_mu_gpytorch_gt,
                pred_sigma_gpytorch_gt,
                pred_mu_gpytorch_ml,
                pred_sigma_gpytorch_ml,
                "Amortized Model",
                "Ground Truth Kernel",
                "Type-2 ML Kernel",
                desciption,
            )
            figure_name = "{}_{}d_prediction_{}".format(prefix, input_dimension, i)
            # add prediction plot to tensorboard
            self.tensorboard_writer.add_figure(figure_name, figure, global_step=epoch_index)
            # except:
            #    print("Error in eval plotter")

    def eval_on_dataset_sample(self, device, model, simulated_datasets_sample, ml_learned_parameters_sample):
        learned_kernel_parameters, learned_noise_variances = ml_learned_parameters_sample
        rmses_ml = []
        rmses_amortized = []
        nlls_ml = []
        nlls_amortized = []
        for i, simulated_dataset in enumerate(simulated_datasets_sample):
            # try:
            assert isinstance(simulated_dataset, SimulatedDataset)
            input_kernel_list = simulated_dataset.get_input_kernel_list()
            ml_learned_observation_noise = np.sqrt(learned_noise_variances[i])
            ml_learned_kernel_state_dict = learned_kernel_parameters[i]
            kernel_expression_ml = transform_kernel_list_to_expression(input_kernel_list, add_prior=False)
            kernel_ml = get_gpytorch_kernel_from_expression_and_state_dict(
                kernel_expression_ml, ml_learned_kernel_state_dict, wrap_in_addition=False
            )
            self.gpytorch_model_config.optimize_hps = False

            ml_gyptroch_model_config = deepcopy(self.gpytorch_model_config)
            ml_gyptroch_model_config.initial_likelihood_noise = ml_learned_observation_noise
            gpytorch_model_ml = GPModelPytorch(kernel=kernel_ml, **ml_gyptroch_model_config.dict())
            X_data, y_data = simulated_dataset.get_dataset(normalized_version=self.normalize_datasets)
            X_test, y_test = simulated_dataset.get_test_dataset(normalized_version=self.normalize_datasets)
            gpytorch_model_ml.infer(X_data, y_data)
            # predict with both models
            pred_mu_gpytorch_ml, pred_sigma_gpytorch_ml = gpytorch_model_ml.predictive_dist(X_test)
            pred_mu_model, _, pred_sigma_model = model.predict(X_test, X_data, y_data, input_kernel_list, device=device)

            # RMSES
            rmses_ml.append(calculate_rmse(pred_mu_gpytorch_ml, y_test))
            rmses_amortized.append(calculate_rmse(pred_mu_model, y_test))
            # NLLs
            nlls_ml.append(calculate_nll_normal(y_test, pred_mu_gpytorch_ml, pred_sigma_gpytorch_ml))
            nlls_amortized.append(calculate_nll_normal(y_test, pred_mu_model, pred_sigma_model))
            # except:
            #    print("Error in evalution on simulated datasets")
        return rmses_ml, rmses_amortized, nlls_ml, nlls_amortized

    def move_to_device(self, batch, device):
        (
            kernel_embeddings,
            kernel_mask,
            X_padded,
            y_padded,
            size_mask,
            dim_mask,
            size_mask_kernel,
            N,
            kernel_list,
            mlls_gt,
            log_posterior_density_gt,
            gt_kernel_parameter_lists,
            observation_noise_gt,
        ) = batch
        kernel_embeddings = kernel_embeddings.to(device)
        kernel_mask = kernel_mask.to(device)
        X_padded = X_padded.to(device)
        y_padded = y_padded.to(device)
        size_mask = size_mask.to(device)
        dim_mask = dim_mask.to(device)
        size_mask_kernel = size_mask_kernel.to(device)
        observation_noise_gt = observation_noise_gt.to(device)
        N = N.to(device)
        return (
            kernel_embeddings,
            kernel_mask,
            X_padded,
            y_padded,
            size_mask,
            dim_mask,
            size_mask_kernel,
            N,
            kernel_list,
            mlls_gt,
            log_posterior_density_gt,
            gt_kernel_parameter_lists,
            observation_noise_gt,
        )

    def standard_GP_learning(self, simulated_dataset_objects: List[SimulatedDataset], add_prior: bool = False):
        kernel_state_dicts = []
        noise_variances = []
        for dataset_object in simulated_dataset_objects:
            kernel_list = dataset_object.get_input_kernel_list()
            kernel_expression = transform_kernel_list_to_expression(kernel_list, add_prior)
            X_data, y_data = dataset_object.get_dataset(self.normalize_datasets)
            self.gpytorch_model_config.optimize_hps = True
            gpytorch_model = GPModelPytorch(kernel=kernel_expression.get_kernel(), **self.gpytorch_model_config.dict())
            gpytorch_model.infer(X_data, y_data)
            kernel_state_dict = gpytorch_model.kernel_module.state_dict()
            kernel_state_dicts.append(kernel_state_dict)
            noise_variance = gpytorch_model.get_likelihood_noise_variance()
            noise_variances.append(noise_variance)
        self.gpytorch_model_config.optimize_hps = False
        return kernel_state_dicts, noise_variances

    def print_shapes(self, X_padded, y_padded, N, size_mask, dim_mask, kernel_embeddings, kernel_mask, size_mask_kernel):
        print("X padded shape")
        print(X_padded.shape)
        print("Y-padded shape")
        print(y_padded.shape)
        print("N shape")
        print(N.shape)
        print("Size mask shape")
        print(size_mask.shape)
        print("Dim mask shape")
        print(dim_mask.shape)

    def create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save_config(self, training_config_dict, model_config_dict, dataset_folder, run_folder, generator_config_dict=None):
        config_dict = training_config_dict
        config_dict["dataset_folder"] = dataset_folder
        config_dict["model_config"] = model_config_dict
        config_dict["generator_config"] = generator_config_dict
        config_file = os.path.join(run_folder, "config.json")
        write_dict_to_json(config_dict, config_file)

    def show_cuda_memory_info(self):
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.total_memory: %fGB" % (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))
        print(torch.cuda.memory_summary())

    def save_checkpoint(self, model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch_index, global_step, folder):
        checkpoint = {}
        checkpoint["model"] = model_state_dict
        checkpoint["optimizer"] = optimizer_state_dict
        checkpoint["learning_rate_scheduler"] = scheduler_state_dict
        checkpoint["epoch_index"] = epoch_index
        checkpoint["global_step"] = global_step
        checkpoint["model_config_dict"] = self.model_config.dict()
        checkpoint["training_config_dict"] = self.training_config.dict()
        file_name = os.path.join(folder, "checkpoint_{}.pth".format(epoch_index))
        torch.save(checkpoint, file_name)

    def load_checkpoint(self, checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        model_state_dict = checkpoint["model"]
        optimizer_state_dict = checkpoint["optimizer"]
        scheduler_state_dict = checkpoint["learning_rate_scheduler"]
        epoch_index = checkpoint["epoch_index"]
        global_step = checkpoint["global_step"]
        model_config_dict = checkpoint["model_config_dict"]
        training_config_dict = checkpoint["training_config_dict"]
        return (
            model_state_dict,
            optimizer_state_dict,
            scheduler_state_dict,
            epoch_index,
            global_step,
            model_config_dict,
            training_config_dict,
        )

    def save_model(self, model_state_dict, epoch_index, folder):
        file_name = os.path.join(folder, "model_snapshot_{}.pth".format(epoch_index))
        torch.save(model_state_dict, file_name)

def string_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def string2bool(b):
    if isinstance(b, bool):
        return b
    if b.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif b.lower() in ("no", "false", "f", "n", "0"):
        return False

def get_checkpoint_enum_from_string(string):
    for e in CheckpointStrategy:
        if string==e.value:
            return e

def parse_args():
    parser = argparse.ArgumentParser(
        description="This is a script for training the amortization neural network for GP parameters with structured kernels"
    )
    parser.add_argument("--checkpoint_file_path",default="")
    parser.add_argument("--output_folder")
    parser.add_argument("--run_name",default="paper_run")
    parser.add_argument("--generator_config_class",default = "DimWiseAdditiveWithNoiseMixedNoMaternBiggerConfig")
    parser.add_argument("--amortized_model_config_class",default="SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig")
    parser.add_argument("--training_config_class", default="BasicDimWiseAdditiveKernelTrainingConfig")
    parser.add_argument("--from_checkpoint", default=False)
    parser.add_argument("--check_point_strategy", default="RESUME_MODEL")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    start_from_checkpoint = string2bool(args.from_checkpoint)

    checkpoint_file_path = args.checkpoint_file_path

    output_folder = args.output_folder
    run_suffix = args.run_name
    seed = 200
    np.random.seed(seed)
    generator_config = string_to_class(args.generator_config_class)()
    generator_config.num_train_datasets_on_the_fly = 2000
    generator_config.num_val_datasets_on_the_fly = 200
    generator_config.num_test_datasets_on_the_fly = 200

    amortized_model_config = string_to_class(args.amortized_model_config_class)()

    training_config = string_to_class(args.training_config_class)()

    trainer = Trainer(
        amortized_model_config,
        "",
        output_folder,
        training_config,
        run_suffix,
        True,
        generator_config,
    )
    trainer.initial_seed = seed
    trainer.checkpoint_strategy = get_checkpoint_enum_from_string(args.check_point_strategy)
    use_gpu = torch.cuda.is_available()
    assert use_gpu
    trainer.train(use_gpu, start_from_checkpoint, checkpoint_file_path)
