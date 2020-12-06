# Text Classification Competition: Twitter Sarcasm Detection

This course project is an entry into the CS 410 Fall 2020 text classification competition. More details about the competition and the relevant datasets can be found at https://github.com/CS410Fall2020/ClassificationCompetition.

## Implementation details

This project utilizes a BERT (Bidirectional Encoder Representations from Transformers) model to classify the data. BERT is a recent NLP language model developed by Google in 2018. The specific BERT model fine-tuned in this project can be found [here](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3).

The classifier model constructed in this project consists of a BERT preprocessing layer, followed by the BERT model, a Dropout layer, and a Dense layer. It uses the response tweet and last context tweet as inputs. This was done for simplicity and also because, intuitively, the tweet that the response tweet is a direct response to is the most relevant context for determining sarcasm. No preprocessing was performed on the text data itself prior to input to the BERT preprocessing model (e.g. stop word removal, stemming, etc.). The model was trained using a sparse categorical cross-entropy loss function and an AdamW optimizer. This model proved sufficient to beat the baseline provided by the competition, so no other models or methods were tried. Minimal hyperparameter tuning was performed, with only the number of training epochs being adjusted for best results and to avoid overfitting.

The code for this project predominantly follows the TensorFlow tutorial found [here](https://www.tensorflow.org/tutorials/text/solve_glue_tasks_using_bert_on_tpu) for using BERT on TPU in Google Colab, with modifications made to accomodate for the structure of the Twitter dataset.

## Utility testing methods

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

## Usage examples

The following code snippets provide a few examples for running the trained classifier on the Twitter sarcasm test set.

**Loading the trained model**

```python
load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
reloaded_model = tf.saved_model.load(saved_model_path, options=load_options)
```

**Downloading and preprocessing the test dataset**

```python
test_url = 'https://raw.githubusercontent.com/CS410Fall2020/ClassificationCompetition/main/data/test.jsonl'
test_path = tf.keras.utils.get_file('test.jsonl', test_url)

test_dataset = preprocess_dataset(test_path, 'test')
```

**Printing some classifier results on the test set**

```python
for test_row in test_dataset.shuffle(1800).map(prepare).take(5):
    print_result(test_row, reloaded_model)
```
```
context: [b'@USER @USER @USER It \xe2\x80\x99 s obvious I \xe2\x80\x99 m dealing with a double digit IQ . Have a good life .']
response: [b'@USER @USER @USER Hahahahahah What a chump . No testicular fortitude at all . It \xe2\x80\x99 s unsurprising that liberals lose with people like this . <URL>']
prediction: SARCASM

context: [b'@USER @USER asked me to respond to @USER . See attached . Thanks for the opportunity . #KXL <URL>']
response: [b'@USER @USER @USER Imagine that . A politician making baseless accusations . Because has * never * done that before .']
prediction: SARCASM

context: [b'@USER @USER By all means you should initiate another failed impeachment , causing further embarrassment ( if that were even possible ) to your party , then go tear up some official documents like a toddler .']
response: [b'@USER @USER @USER Yet you have no shame in supporting the biggest criminal in White House history .']
prediction: SARCASM

context: [b'@USER @USER Aaaayyyyyeeee I \xe2\x80\x99 m the Hypemobile humie . I gatchu ! ! ! \xf0\x9f\x98\x9c \xf0\x9f\x91\x8c \xf0\x9f\x8f\xbc']
response: [b'@USER @USER Oh ... OHHHH ! That \xe2\x80\x99 s how it \xe2\x80\x99 s gonna be ? My two bestfriends just gon \xe2\x80\x99 team up on me ? \xf0\x9f\x98\xa1']
prediction: NOT_SARCASM

context: [b"@USER @USER @USER Thank You so much , Diablo , My Dearest Friend ! I am grateful every day , as I am happy every day . I make those choices every day for me ! It doesn't matter what is going on , my choices stick . I give the gift of positivity to myself , to everyone I touch ! LOVE U XOXO <URL>"]
response: [b'@USER @USER @USER All good things are possible when the #heart illuminates the mind #ThinkBIGSundayWithMarsha #InspireThemRetweetTuesday <URL>']
prediction: NOT_SARCASM
```
