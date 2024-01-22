################# Section 1: Libraries & Configurations #################
from transformers import AutoModel, AutoTokenizer
import tensorflow_datasets as tfds
from transformers import BertTokenizer, TFBertModel, RobertaTokenizer, TFRobertaModel, DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
import optuna

import argparse
import logging
import sys
from tqdm import tqdm
import random

tf.random.set_seed(42)
random.seed(42)

parser = argparse.ArgumentParser(description='Process IMDB dataset and generate BERT embeddings.')
parser.add_argument('--testing', action='store_true', default=True, help='Use a smaller subset of the dataset for testing purposes.')
args = parser.parse_args()
testing = args.testing

production = True
if production:
    testing = False


if testing:
    loglevel = logging.DEBUG
    n_trials = 2
    epochs_per_model = 3
else:
    loglevel = logging.INFO
    n_trials = 100
    epochs_per_model = 50
batch_size = 5
es_patience = 5
logging.basicConfig(level=loglevel,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    ) 


################# Section 2: Loading in Dataset & Models #################
dataset_size = '60%' if not testing else '.1%'  # Use only 1% of the dataset if testing
train_data = tfds.load('imdb_reviews', split=f'train[:{dataset_size}]', as_supervised=True)

tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer_roberta = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
model_roberta = TFRobertaModel.from_pretrained("cardiffnlp/twitter-roberta-base")
tokenizer_distilbert = DistilBertTokenizer.from_pretrained("huggingface-course/distilbert-base-uncased-finetuned-imdb")
model_distilbert = TFDistilBertModel.from_pretrained("huggingface-course/distilbert-base-uncased-finetuned-imdb")


################# Section 3: Embedding the Dataset (Features) #################
def generate_embeddings(text, model_name):
    logging.debug("Generating embeddings...")

    # Use the global model reference based on the model name
    if model_name == 'bert':
        model = model_bert
        tokenizer = tokenizer_bert
    elif model_name == 'roberta':
        model = model_roberta
        tokenizer = tokenizer_roberta
    elif model_name == 'distilbert':
        model = model_distilbert
        tokenizer = tokenizer_distilbert

    
    if isinstance(text, tf.Tensor):
        text = text.numpy()  # Convert tensor to numpy array

    if isinstance(text, bytes):
        text = text.decode('utf-8')

    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :]

embeddings = []
labels = []

logging.info("Processing each review in the training dataset...")
for text, label in tqdm(train_data, desc="Processing Reviews"):
    bert_embeddings = tf.py_function(generate_embeddings, [text, 'bert'], Tout=tf.float32)
    roberta_embeddings = tf.py_function(generate_embeddings, [text, 'roberta'], Tout=tf.float32)
    distilbert_embeddings = tf.py_function(generate_embeddings, [text, 'distilbert'], Tout=tf.float32)

    concatenated_embeddings = tf.concat([bert_embeddings, roberta_embeddings, distilbert_embeddings], axis=1)
    concatenated_embeddings = tf.squeeze(concatenated_embeddings, axis=0) 
    logging.debug(f"Concatenated embeddings size: {concatenated_embeddings.shape}")
    logging.debug(f"Label: {label}")
    embeddings.append(concatenated_embeddings)
    labels.append(label)

embeddings = tf.convert_to_tensor(embeddings)
labels = tf.convert_to_tensor(labels)
print("Embeddings shape", embeddings.shape)
print("Labels Shape", labels.shape)

################# Section 4: Prepping Data for Training #################
dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)


################# Section 5: Model Architecture #################
def generate_binary_mlp(num_layers:int, initial_neurons:int,  dropout_rate:float, activation_function:str, scaling_factor:float):
    '''scaling factor - between 0 and 1.'''
    model = tf.keras.Sequential()

    for i in range(num_layers):
        # For the first layer, add the specified number of neurons
        if i == 0:
            model.add(Dense(initial_neurons, activation=activation_function))
        else:
            # Decrease the number of neurons by the scaling factor
            neurons = int(initial_neurons * (scaling_factor ** i))
            model.add(Dense(neurons, activation=activation_function))

        # Add dropout after each Dense layer
        model.add(Dropout(dropout_rate))

    # Add the final layer with output_classes neurons
    model.add(Dense(1, activation='sigmoid'))
    logging.debug("MLP Layer Completed.")

    return model

################# Section 6: Preparing Hyperparameter Tuning #################
logging.info("starting model training")
early_stopping = EarlyStopping(monitor='accuracy', patience=es_patience, verbose=1)

def objective(trial):
    # Define the hyperparameters
    num_layers = trial.suggest_int('num_layers', 2, 10)
    initial_neurons = trial.suggest_int('initial_neurons', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'tanh', 'sigmoid', 'softmax'])
    scaling_factor = trial.suggest_float('scaling_factor', 0.5, 1.0)

    # Create, compile and train the model
    model = generate_binary_mlp(num_layers, initial_neurons, dropout_rate, activation_function, scaling_factor)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

    # Train the model
    history = model.fit(dataset, epochs=epochs_per_model, verbose=0, callbacks=[early_stopping])  # Adjust epochs as needed
    logging.info(f'{history.history}')
    # Return the evaluation metric value
    return max(history.history['accuracy']) 


################# Section 7: Hyperparameter Tuning or Training best Model#################
if not production:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    logging.info(f'Best hyperparameters: {study.best_trial.params}')

################# Section 8: Saving the best Model #################
# Recreating best model
if production:
    # [I 2023-12-14 17:21:20,892] Trial 34 finished with value: 0.9332000017166138 and parameters: 
    # {'num_layers': 2, 'initial_neurons': 149, 'dropout_rate': 0.3971740887697476, 'activation_function': 'sigmoid', 'scaling_factor': 0.5506064499981852}. 
    # Best is trial 34 with value: 0.9332000017166138.
    num_layers = 2
    initial_neurons = 149
    dropout_rate = .4
    activation = "sigmoid"
    scaling_factor = .55

    best_model = generate_binary_mlp(num_layers, initial_neurons, dropout_rate, activation, scaling_factor)
else: 
    best_trial = study.best_trial
    
    best_model = generate_binary_mlp(num_layers=best_trial.params['num_layers'],
                                    initial_neurons=best_trial.params['initial_neurons'],
                                    dropout_rate=best_trial.params['dropout_rate'],
                                    activation_function=best_trial.params['activation_function'],
                                    scaling_factor=best_trial.params['scaling_factor'])

best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

best_model.fit(dataset, epochs=epochs_per_model, verbose=1)  
best_model.save('savedModel.h5') 