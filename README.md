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
context: [b'@USER How can we get promotions committees to rethink some of this ? #mededchat']
response: [b'@USER @USER Yes of course you can . But probably need to do this by publishing papers .. lol . Check out the work by @USER et al <URL> et al <URL> And group : <URL> Turn into white paper for P & T cmte !']
prediction: NOT_SARCASM

context: [b'It was inappropriate for the president to ask a foreign leader to investigate his political opponent and to withhold United States aid to encourage that investigation . 5/15']
response: [b'@USER When our election is further compromised by foreign entanglements & degrades American voter confidence in our democratic constitutional republic , this decision of yours will stain your reputation forever .']
prediction: NOT_SARCASM

context: [b"@USER @USER @USER Just another example of ' s FUBAR . Just got this message . You've got some bugs . Start with freeing and everyone from Twitter jail . #HeroesResist #BringBackHeroes <URL>"]
response: [b'@USER @USER @USER Ridiculous . I have been following for years . He has always been incredibly decent and respectful . But T gets to say and do whateva , \xe2\x9c\x8c \xf0\x9f\x8f\xbe BS .']
prediction: NOT_SARCASM

context: [b'@USER @USER Hey Ash here . Selling or #Service ? Tell me you both ! \xf0\x9f\x92\x99']
response: [b'@USER @USER Correction : Selling and #Service but for free ? Tell me guys ? But yeah let \xe2\x80\x99 s for a while #TakeOver and later once our goals are achieved we \xe2\x80\x99 ll leave it to the . What say ? \xf0\x9f\x92\x99']
prediction: NOT_SARCASM

context: [b'@USER @USER I \xe2\x80\x99 m not in position to say that . I \xe2\x80\x99 m simply a concerned , loyal and patriotic citizen who wants and believes in his country and crave for everything to work properly .']
response: [b'@USER @USER @USER Suffice to say , there\'s no basis for & there should be NO investigation . After all , for d love of d country , one\'s OWN daughter that is " eminently qualified " can & should be employed & appointed to work & " serve " d citizenry . Wailers see " patriotic " act as NEPOTIC act']
prediction: SARCASM
```
