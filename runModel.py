################# Section 1: Libraries & Configurations #################
# Include all necessary imports as in your original script
from transformers import BertTokenizer, TFBertModel, RobertaTokenizer, TFRobertaModel, DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import logging

parser = argparse.ArgumentParser(description='Process IMDB dataset and generate BERT embeddings.')
parser.add_argument('--testing', action='store_true', default=True, help='Use a smaller subset of the dataset for testing purposes.')
args = parser.parse_args()
testing = args.testing

# Configurations
tf.random.set_seed(42)

################# Section 2: Load Model #################
model = tf.keras.models.load_model('savedModel.h5')

################# Section 3: Load Test Data #################
# Load only the test part of the dataset
dataset_size = '100%' if not testing else '1%'  # Use only 1% of the dataset if testing
# train_data = tfds.load('imdb_reviews', split=f'train[:{dataset_size}]', as_supervised=True)
test_data = tfds.load('imdb_reviews',  split=f'test[:{dataset_size}]', as_supervised=True)


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
for text, label in tqdm(test_data, desc="Processing Reviews"):
    bert_embeddings = tf.py_function(generate_embeddings, [text, 'bert'], Tout=tf.float32)
    roberta_embeddings = tf.py_function(generate_embeddings, [text, 'roberta'], Tout=tf.float32)
    distilbert_embeddings = tf.py_function(generate_embeddings, [text, 'distilbert'], Tout=tf.float32)

    concatenated_embeddings = tf.concat([bert_embeddings, roberta_embeddings, distilbert_embeddings], axis=1)
    concatenated_embeddings = tf.squeeze(concatenated_embeddings, axis=0) 
    logging.debug(f"Concatenated embeddings size: {concatenated_embeddings.shape}")
    logging.debug(f"Label: {label}")
    embeddings.append(concatenated_embeddings)
    labels.append(label)

test_embeddings = tf.convert_to_tensor(embeddings)
test_labels = tf.convert_to_tensor(labels)


################# Section 5: Evaluate Model #################
predicted_probs = model.predict(test_embeddings)
predicted_labels = (predicted_probs > 0.5).astype("int32")

################# Section 6: Print Results #################
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

# Print Accuracy, Precision, Recall, and F1-Score
results = f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n'

# Calculate and Print Confusion Matrix
confusion = confusion_matrix(test_labels, predicted_labels)
results += 'Confusion Matrix:\n' + str(confusion) + '\n'

# Writing results to a text file
results_file_path = '/mnt/data/model_performance.txt'
with open(results_file_path, 'w') as file:
    file.write(results)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save the plot to a PNG file
confusion_matrix_file_path = '/mnt/data/confusion_matrix.png'
plt.savefig(confusion_matrix_file_path)

results_file_path, confusion_matrix_file_path