# CNN Experiments

This repository contains Python modules and resources for training and evaluating CNN architectures, including conventional CNNs and custom architectures for few-shot learning tasks.

## Reproducing Experiments

All experiments mentioned in the associated report can be reproduced using the provided Python modules:

1. **`training.py`**: For training conventional CNN architectures.
2. **`fewshot_training.py`**: For episode training using a custom architecture tailored for few-shot learning tasks.

### Dataset Structure

The modules assume the datasets are organized in the working directory as follows:
```
./train
./valid
./test
```

### Training Outputs

Trained models and their progress will be automatically saved in the working directory. The file identifier for saved models is determined by the `model_name` parameter.

## Configuration

Arguments defining the configuration of the training process are detailed below:

### Configuration Parameters for `training.py` Module

| **Argument**        | **Default Value** | **Description**                                                                 |
|----------------------|-------------------|---------------------------------------------------------------------------------|
| `architecture`      | custom            | Model architecture: [custom, efficientnet, resnet, densenet]                   |
| `lr`                | 0.001             | Learning rate                                                                  |
| `batch_size`        | 64                | Batch size                                                                     |
| `weight_decay`      | 0.0               | Weight decay                                                                   |
| `optimizer`         | adam              | Optimizer name                                                                 |
| `dropout`           | 0                 | Dropout rate                                                                   |
| `num_trainings`     | 3                 | Number of training runs                                                        |
| `num_epochs`        | 10                | Number of epochs per run                                                       |
| `model_name`        | tmp               | Model name identifier                                                          |
| `double_augment`    | 0                 | Use double data with one augmented segment [1/0]                               |
| `fraction`          | 1.0               | Fraction of the training data to be used                                       |
| `augmentations`     | `[]`              | List of augmentations to apply                                                 |

### Configuration Parameters for `fewshot_training.py` Module

| **Argument**        | **Default Value** | **Description**                                                                 |
|----------------------|-------------------|---------------------------------------------------------------------------------|
| `num_classes`       | 5                 | Number of classes (n-way)                                                      |
| `cnn_channels`      | 32                | Number of CNN output channels                                                  |
| `embed_dim`         | 64                | Embedding dimension                                                            |
| `num_heads`         | 2                 | Number of attention heads                                                      |
| `patch_size`        | 2                 | Patch size for Vision Transformer                                              |
| `k_shot`            | 5                 | Number of shots (examples per class)                                           |
| `q_query`           | 15                | Number of query samples per class                                              |
| `fraction`          | 0.01              | Fraction of the dataset to use                                                 |
| `num_trainings`     | 3                 | Number of training runs                                                        |
| `num_episodes`      | 3000              | Number of episodes per training run                                            |
| `model_name`        | 'tmp'             | Model name identifier                                                          |
| `lr`                | 0.001             | Learning rate                                                                  |
| `optimizer`         | 'adam'            | Optimizer name                                                                 |
| `weight_decay`      | 0.0               | Weight decay                                                                   |

## Requirements

All necessary libraries are listed in the `requirements.txt` file. Please ensure you install them before running the modules:
```bash
pip install -r requirements.txt
```

## Usage

To train a model, run the respective module with the desired configuration:
```bash
python training.py --arg1 value1 --arg2 value2
```
or
```bash
python fewshot_training.py --arg1 value1 --arg2 value2
```

Refer to the report for detailed descriptions of the arguments and their usage.

### Initializing Training

Provided that a Python environment is available, a simple command of:
```bash
pip install -r requirements.txt
```
followed by:
```bash
python training.py
```
will initialize a training process.

