# DPFairDeepLearning

This repository contains code to implement differentially private fair deep learning models for tabular data. The code allows training models with different architectures, datasets, and privacy guarantees.

## Models
The following model architectures are implemented:

- DemParModel - Implements statistical parity via an adversarial network
- EqualOddModel - Extends DemParModel for equalized odds via additional input to adversary

The models consist of an autoencoder, classifier, and adversary. Differential privacy can be applied to the encoder+classifier or adversary components.

## Datasets
The following datasets are used:

- Adult 
- German
- CelebA

Preprocessing code is provided for each dataset to extract features, labels, and sensitive attributes.

## Training
The main training code is in **run_training.py**. This handles model instantiation, data loading, and trainer initialization.

The Trainer class in **trainer.py** handles the full training loop including:

- Making models differentially private
- Training autoencoder + classifier
- Adversarial training loop 
- Logging metrics
- Evaluating fairness metrics
- Checkpointing based on accuracy and fairness

The training_script.py shows how the hyperparameters and model configurations are defined. It generates the full set of experiments and executes them, leveraging parallel threads.

## Utilities

**utils.py** contains utility functions like:

- Converting data to torch Tensors 
- Generating config file strings for ClearML
- Logging metrics to ClearML
- Filtering hyperparameter combinations

## Notebooks

The **data_management** folder contains Jupyter notebooks for saving/loading experiments from ClearML.

## Requirements

The code requires Python 3 and the following libraries:

- PyTorch
- ClearML
- Scikit-learn
- Numpy
- Pandas

Install requirements with:

```
pip install -r requirements.txt
```

## Running Experiments 

To run an example experiment:

```bash
python run_training.py --dataset Adult --model DP --epochs 10 
```

This will train a DemPar model on the Adult dataset for 10 epochs.

See **training_script.py** for the full list of possible hyperparameters.

## Instructions

To run an experiment:

1. Call **run_training.py** for each desired configuration or generate and execute list of hyperparameter configurations in **training_script.py**
2. Monitor experiments in ClearML Web UI  
3. Use notebooks to collect results

So in summary, the key files are:

- **run_training.py** - Main training loop
- **trainer.py** - Full training implementation  
- **training_script.py** - Hyperparameter generation and execution main loop
- **utils.py** - Helper functions
- **data_management/** - Notebooks for ClearML

The other files provide dataset processing, model architectures, metrics, etc.
