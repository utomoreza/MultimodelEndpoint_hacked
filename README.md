# GPU-based 'multi-model' endpoint on SageMaker

## Introduction

This repo contains all files and folders needed to manipulate SageMaker inference endpoint so that one can have GPU-based 'multi-model' endpoint. Please read my two articles on Medium [here](https://utomorezadwi.medium.com/deploying-gpu-based-models-on-sagemaker-using-multi-model-endpoint-part-1-da68cbbf3d04) and [here](https://utomorezadwi.medium.com/deploying-gpu-based-models-on-sagemaker-using-multi-model-endpoint-part-2-final-6e05cf10142f) which explain the concept in more details.

## Requirements

In order to run this repo, you need to install the following dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `torchvision`
- `tensorflow`
- `boto3`
- `sagemaker`

In addition, you are recommended to install AWS CLI from [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

## Structure

The repo is structured as follows:
- [`model`](./model): the main directory to be compressed into `model.tar.gz` file
- [`train_imdb_tf`](./train_imdb_tf): the directory containing training process for the TensorFlow model
- [`train_mnist_pt`](./train_mnist_pt): the directory containing training process for the PyTorch model
- [`deploy.ipynb`](./deploy.ipynb): the main for doing deployment of the models
- [`invoke.ipynb`](./invoke.ipynb): the notebook for invoking the deployed endpoint
- [`debug_pt.py`](./debug_pt.py): the Python script for debugging the PyTorch model
- [`debug_tf.py`](./debug_tf.py): the Python script for debugging the TensorFlow model

## License

This repo is licensed under [MIT License](./LICENSE).
