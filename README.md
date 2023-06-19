# Amor-Struct-GP 
This repo contains the associated code for the paper "Amortized Inference for Gaussian Process Hyperparameters of Structured Kernels" (UAI 2023). In summary, in the paper, we propose predicting the hyperparameters of a Gaussian Process for a given dataset and a configured (input) kernel structure - instead of optimizing/learning them. This drastically speeds up GP inference. In particular, given a training dataset $D_{tr}$ and a kernel given as a symbolic description (e.g. $\mathcal{S}=SE + LIN$ for a sum of a linear kernel and a SE kernel ), we use our neural network $g_{\psi}$ to predict GP parameters
$$\phi_{\mathcal{S}}=g_{\psi}(D_{tr},\mathcal{S})$$
The GP parameters can then be plugged into the analytical/closed-form predictive distribution of the GP $p(y|x,D_{tr},\phi_{\mathcal{S}})$  to get a prediction at input $x$. Our main contribution is that the kernel can be given as input to the neural network and the amortization network does not need to be retrained for a different kernel. Thus the code/method can be used to replace large parts of a GP library via a neural network - as it can be used to replace learning of GP's for a large set of possible kernels.

## Setup

After cloning the repo or extracting the repo from a zip file, first switch to the base folder where this README lies and build the `conda` environment and the package itself via
```buildoutcfg
conda env create --file environment.yml
conda activate amorstructgp
pip install -e .
```

## Perform Tests

To see if everything is set up correctly perform the tests from the repo folder via
```
pytest
```

## Usage of trained model
We provide two pretrained versions of the amortization neural network [here](https://github.com/boschresearch/Amor-Struct-GP-pretrained-weights). We provide a version from the paper, where we use as base kernels SE, LIN, PER and its two gram multiplications. We furthermore provide and a version with Matern52 kernel (and the associcated multiplications) included. We give an example on how to use the model for end-to-end prediction in `prediction_example.ipynb` notebook in the folder `notebooks`.

## Train model 
We provide a training script to train the amortization network from scratch. This can be used for example for training of bigger models or on more synthetic datasets. For training we need three configuration objects. First we need a child of a `BasicAmortizedInferenceModelConfig` configuration, which speficies the architecture of the amortization model (can be found in `config/nn/amortized_infer_models_configs.py`). Secondly, we need to configure the training data sampler via a child of `BasicDimWiseAdditiveGeneratorConfig` configuration (can be found in `config/data_generators/dim_wise_additive_generator_config.py`).  Third, we need to configure training settings. Here, we need to use a child of `BasicDimWiseAdditiveKernelTrainingConfig` (can be found in `config/training/training_configs.py`). Furthermore we need a output path `OUTPUT_PATH` and path to a checkpoint file `CHECKPOINT_PATH` in case training should be resumed. For the standard training in the paper we can than execute from the `training` folder (in a GPU environment)
```buildoutcfg
python trainer.py --output_folder=OUTPUT_PATH --run_name=main_run --generator_config_class=DimWiseAdditiveWithNoiseMixedNoMaternBiggerConfig --amortized_model_config_class=SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig --training_config_class=BasicDimWiseAdditiveKernelTrainingConfig
```
For noise-variance finetuning after the first training phase (suppose we stored a checkpoint in `CHECKPOINT_PATH`) we can use
```buildoutcfg
python trainer.py --output_folder=OUTPUT_PATH --run_name=noise_fine_tuning --from_checkpoint=True --checkpoint_file_path=CHECKPOINT_PATH --generator_config_class=DimWiseAdditiveWithNoiseMixedNoMaternBiggerConfig --amortized_model_config_class=SmallerStandardNoDropoutCrossAttentionKernelEncSharedDatasetEncMLPWrapperAmortizedModelConfig --training_config_class=DimWiseAdditiveKernelFineTuningConfig
```

For using a different architecture (for example with more parameters) we recommend using and changing the `ExperimentalAmortizedModelConfig`. This is also automatically used in the configuration `ExperimentalAmortizedStructuredConfig` which is a child of `BasicGPModelAmortizedStructuredConfig` (see `config/models/gp_model_amortized_structured_config.py`) which is used to build the final wrapped `BaseModel` child instance that is used for employing the amortization model for inference (see `prediction_example.ipynb`).

Parameter decription of `trainer.py`:

`--output_folder`: Folder where checkpoints and tensorboard files are stored.

`--run_name`: Suffix that is used to identify runs

`--generator_config_class`: name of generator config class (any class in `config/data_generators/dim_wise_additive_generator_config.py`). Configures the sampling distribution - including dataset sizes of the training datasets, dimension sizes, if matern should be included or not or if only positive examples should be sampled

`--amortized_model_config_class`: name of amortized model config class (any class in `config/nn/amortized_infer_models_configs.py`). Configures the architecture of the amortized model including how many layers etc.

`--training_config_class`: name of training config class (any class in `config/training/training_configs.py`). Configures standard training settings like number of epochs (each epoch conists of 2000 newly generated datasets - 4500 epochs = 9 Mio datasets), batch size, optimizer and loss function type.

`--from_checkpoint`: flag if training should be continued from checkpoint. How it deals with the checkpoint (if optimizer is also restored) is give via the `check_point_strategy` flag.

`--check_point_strategy`: can take three inputs: RESUME_COMPLETE, RESUME_MODEL_AND_OPT, RESUME_MODEL - RESUME_COMPLETE continues the training from the exact state where it was finished, RESUME_MODEL_AND_OPT restores model and optimizer, RESUME_MODEL (default) resumes only the model.


## License

Amor-Struct-GP is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.