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
from copy import deepcopy
from amorstructgp.config.models.base_model_config import BaseModelConfig
from amorstructgp.config.models.gp_model_amortized_ensemble_config import BasicGPModelAmortizedEnsembleConfig
from amorstructgp.config.models.gp_model_amortized_structured_config import BasicGPModelAmortizedStructuredConfig
from amorstructgp.gp.gpytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from amorstructgp.models.gp_model_amortized_structured import GPModelAmortizedStructured
from amorstructgp.config.models.gp_model_gpytorch_config import BasicGPModelPytorchConfig
from amorstructgp.models.gp_model_pytorch import GPModelPytorch
from amorstructgp.models.gp_model_amortized_ensemble import GPModelAmortizedEnsemble
import numpy as np


class ModelFactory:
    @staticmethod
    def build(model_config: BaseModelConfig):
        if isinstance(model_config, BasicGPModelPytorchConfig):
            kernel = PytorchKernelFactory.build(model_config.kernel_config)
            model = GPModelPytorch(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicGPModelAmortizedStructuredConfig):
            kernel_config = model_config.kernel_config
            kernel_list = kernel_config.kernel_list
            model = GPModelAmortizedStructured(
                model_config.prediction_quantity,
                model_config.amortized_model_config,
                model_config.do_warm_start,
                model_config.warm_start_steps,
                model_config.warm_start_lr,
            )
            model.set_kernel_list(kernel_list)
            model.load_amortized_model(model_config.checkpoint_path)
            return model

        elif isinstance(model_config, BasicGPModelAmortizedEnsembleConfig):
            kernel_list = model_config.kernel_list
            model = GPModelAmortizedEnsemble(
                model_config.prediction_quantity, model_config.amortized_model_config, model_config.entropy_approximation
            )
            model.set_kernel_list(kernel_list)
            model.load_amortized_model(model_config.checkpoint_path)
            return model
        else:
            raise NotImplementedError(f"Invalid config: {model_config.__class__.__name__}")

    @staticmethod
    def change_input_dimension(model_config: BaseModelConfig, input_dimension=int) -> BaseModelConfig:
        transformed_model_config = deepcopy(model_config)
        if hasattr(transformed_model_config, "input_dimension"):
            transformed_model_config.input_dimension = input_dimension
        if hasattr(transformed_model_config, "kernel_config") and hasattr(transformed_model_config.kernel_config, "input_dimension"):
            transformed_model_config.kernel_config.input_dimension = input_dimension
        return transformed_model_config


if __name__ == "__main__":
    pass
