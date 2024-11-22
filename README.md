# IMDB Sentiment Analysis with DistilBERT

This project demonstrates fine-tuning DistilBERT for sentiment analysis on the IMDB movie review dataset.

## What is DistilBERT?

DistilBERT is a smaller, faster, and lighter version of the BERT (Bidirectional Encoder Representations from Transformers) language model. It is trained using a technique called knowledge distillation, which allows it to retain most of BERT's performance while being significantly more efficient. DistilBERT has 40% fewer parameters than BERT but retains 97% of its language understanding capabilities. It is well-suited for tasks like sentiment analysis, where computational resources might be limited.

## Running the Code

This code is designed to be run in a Google Colab environment. Here's how you can access and run it:

1. **Open the notebook:**
   - Click on the provided link to the Colab notebook. (You would need to provide a link to your Colab notebook here).

2. **Connect to a runtime:**
   - In Colab, click on "Connect" in the top right corner to connect to a runtime environment.

3. **Run the cells:**
   - The notebook is divided into cells containing code and markdown. To run a cell, click on the play button next to the cell or use the keyboard shortcut `Shift + Enter`.

4. **Follow the instructions:**
   - The notebook includes comments and markdown cells that provide instructions and explanations. Follow these instructions to run the code and understand the results.

**Note:** Make sure you have the necessary libraries installed. You can install them using the following command in a code cell:
Use code with caution
`bash !pip install datasets transformers[torch] evaluate`

 
This will install the `datasets`, `transformers`, and `evaluate` libraries 

## Code Overview

### 1. Data Loading and Preparation

- The code imports necessary libraries, including `datasets`, `transformers`, `evaluate`, `torch`, and `numpy`.
- The IMDB dataset is loaded using the `load_dataset` function from `datasets`.
- A subset of the training and test data is selected (4000 training samples and 300 test samples) and shuffled for better generalization. This selection is done to reduce the computational burden for demonstration purposes. 
- The `AutoTokenizer` from `transformers` is used to tokenize the text data, converting it into numerical representations that DistilBERT can understand. The `distilbert-base-uncased` tokenizer is used, which means the model is case-insensitive.
- A `DataCollatorWithPadding` is used to handle padding during batching. This ensures that all input sequences have the same length, which is required by the model.

### 2. Model Loading and Fine-tuning

- `AutoModelForSequenceClassification` is used to load the pre-trained DistilBERT model with two output labels for sentiment classification (positive and negative).
- Training arguments are defined using the `TrainingArguments` class. These arguments include:
    - `output_dir`: The directory where training outputs will be saved.
    - `learning_rate`: The rate at which the model learns.
    - `per_device_train_batch_size`: The number of training samples processed in each batch.
    - `per_device_eval_batch_size`: The number of evaluation samples processed in each batch.
    - `num_train_epochs`: The number of times the model is trained on the entire training data.
    - `weight_decay`: A regularization technique to prevent overfitting.
    - `save_strategy`: How often to save model checkpoints.
    - `push_to_hub`: Whether to push the model to the Hugging Face model hub.
- The `Trainer` class from `transformers` is used to fine-tune the model on the training data. This process involves adjusting the model's parameters to improve its performance on the sentiment classification task.
- The model is trained using the `trainer.train()` method.
- The model is evaluated on the test data using the `trainer.evaluate()` method.

### 3. Model Evaluation and Saving

- The model's performance is evaluated using metrics like accuracy, precision, recall, and F1 score. These metrics are computed using the `compute_metrics` function, which utilizes the `evaluate` library.
- The trained model is saved to a file using `torch.save`.

### 4. Sentiment Analysis Pipeline

- A sentiment analysis pipeline is created using the `pipeline` function from `transformers`. This pipeline makes it easy to use the fine-tuned model for making predictions on new text inputs.
- The pipeline is initialized with the task of "sentiment-analysis," the fine-tuned model, and the tokenizer.
- Example usage of the pipeline is shown with the `sentiment_model()` function, which takes a text input and returns a dictionary containing the predicted sentiment label ("NEGATIVE" or "POSITIVE") and its associated score.

## Output

The main output of the code is a fine-tuned DistilBERT model for sentiment analysis. 

- **Evaluation Metrics:** The `trainer.evaluate()` function provides performance metrics on the unseen test data. It typically includes:
    - **Accuracy:** The overall percentage of correctly classified samples.
    - **Precision:** The proportion of true positive predictions among all positive predictions.
    - **Recall:** The proportion of true positive predictions among all actual positive samples.
    - **F1 Score:** The harmonic mean of precision and recall, providing a balanced measure of performance.
- **Sentiment Prediction:** The `sentiment_model()` allows you to predict the sentiment of new text. It returns a dictionary containing:
    - **label:** The predicted sentiment label ("NEGATIVE" or "POSITIVE").
    - **score:** The model's confidence in its prediction, ranging from 0 to 1. A score closer to 1 indicates higher confidence in the predicted label.
