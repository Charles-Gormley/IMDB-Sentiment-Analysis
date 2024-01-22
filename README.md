
# IMDB Review Sentiment Analysis with BERT Embeddings

This project focuses on analyzing sentiment in IMDB movie reviews using embeddings from pre-trained BERT models. The workflow encompasses data preprocessing, model training, hyperparameter tuning, and model evaluation.

## Table of Contents
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- Hugging Face Transformers
- Optuna (for hyperparameter tuning)
- TQDM (for progress bar)

To install these dependencies, run:
```bash
pip install tensorflow transformers optuna tqdm
```

## Dataset
We use the `imdb_reviews` dataset from TensorFlow Datasets. Depending on whether the script is in testing or production mode, a different percentage of the dataset is used.

## Model Architecture
The project utilizes embeddings from three different BERT models:
- BERT Base (Uncased)
- RoBERTa Base
- DistilBERT

These embeddings are concatenated and fed into a custom Multi-Layer Perceptron (MLP) for binary classification (positive or negative review).

## Training Process
1. **Embedding Generation:** The reviews are tokenized and passed through the BERT models to generate embeddings.
2. **Data Preparation:** The embeddings and labels are converted into TensorFlow Datasets.
3. **MLP Training:** A binary MLP is trained on these embeddings.

## Hyperparameter Tuning
Optuna is used for hyperparameter optimization. The script will search for the best combination of:
- Number of layers
- Initial number of neurons
- Dropout rate
- Activation function
- Neuron scaling factor

The search space and trial count can be adjusted based on whether the script is in testing or production mode.

## Model Evaluation
The model's performance is evaluated using accuracy on the test dataset. The script prints out the test score and accuracy upon completion.

## Usage
The script can be run in either testing or production mode. In testing mode, a smaller subset of the dataset is used, and fewer trials are run for hyperparameter tuning. In production mode, the full dataset and a larger number of trials are used.

To run the script, use:
```bash
python script_name.py [--testing]
```
The `--testing` flag is optional and runs the script in testing mode.

---

**Note:** The code saves the best model as 'savedModel.h5', which can be loaded for later use or inference. Ensure TensorFlow is properly installed to load and use the saved model.
