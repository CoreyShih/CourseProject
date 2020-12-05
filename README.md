# Text Classification Competition: Twitter Sarcasm Detection

This course project is an entry into the CS 410 Fall 2020 text classification competition. More details about the competition and the relevant datasets can be found at https://github.com/CS410Fall2020/ClassificationCompetition.

This project utilizes a BERT (Bidirectional Encoder Representations from Transformers) model to classify the data. BERT is a recent NLP language model developed by Google in 2018. The specific BERT model fine-tuned in this project can be found [here](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3). The classifier model constructed in this project consists of a BERT preprocessing layer, followed by the BERT model, a Dropout layer, and a Dense layer. It uses the response tweet and last context tweet as inputs. This was done for simplicity and also because, intuitively, the tweet that the response tweet is a direct response to is the most relevant context for determining sarcasm. The model was trained using a sparse categorical cross-entropy loss function and an AdamW optimizer. This model proved sufficient to beat the baseline provided by the competition, so no other models or methods were tried. Minimal hyperparameter tuning was performed, with only the number of training epochs being adjusted for best results.

The code for this project predominantly follows the TensorFlow tutorial found [here](https://www.tensorflow.org/tutorials/text/solve_glue_tasks_using_bert_on_tpu) for using BERT on TPU in Google Colab, with modifications made to accomodate for the structure of the Twitter dataset.

***Utility testing methods***

Several utility methods are included in the code to assist with running the model on the test dataset and viewing the results.

```python
def preprocess_dataset(dataset_path, split):
    """Processes Twitter sarcasm data into tf.data.Dataset.
    
    Args:
        dataset_path: str path of jsonl dataset.
        split: str designating train or test dataset, either 'train' or 'test'.
    
    Returns:
        A tf.data.Dataset of the Twitter sarcasm data, retaining only the response
        tweet, the last context tweet, and the label if present.
    """
```

```python
def prepare(record):
    """Prepares records from processed dataset for prediction.
    
    Args:
        record: dict of str Tensors.
    
    Returns:
        A list of lists of str Tensors.
    """
```

```python
def get_result(test_row, model):
    """Predicts whether a Twitter sarcasm test example is sarcasm or not sarcasm.
    
    Args:
        test_row: list of str Tensors.
        model: TensorFlow SavedModel for the sarcasm classifier.
    
    Returns:
        A str, either 'SARCASM' or 'NOT_SARCASM', corresponding to the predicted result.
    """
```

```python
def print_result(test_row, model):
    """Prints out the context, response, and predicted label for a Twitter sarcasm test example.
    
    Args:
        test_row: list of str Tensors.
        model: TensorFlow SavedModel for the sarcasm classifier.
    
    Returns:
        None.
    """
```
