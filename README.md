# Restaurant Review Sentiment Analysis

Classify a restaurant review as **positive or negative** using text cleaning, Bag of Words and Gaussian Naive Bayes.

**Python · NLTK · pandas · scikit-learn**

## Files

- [Notebook](natural_language_processing.ipynb)
- [Restaurant_Reviews.tsv](Restaurant_Reviews.tsv): **1,000 reviews**, with `Review` text and a `Liked` label (1 = positive, 0 = negative).

## How it works

1. Remove non-letter characters and convert text to lowercase.
2. Remove common stopwords while keeping `not` and `is`.
3. Apply Porter stemming to reduce words to shorter forms.
4. Split into **800 training and 200 test reviews** (`random_state=0`).
5. Fit CountVectorizer on training reviews only, with a maximum of 1,500 features.
6. Train GaussianNB and evaluate predictions on the test reviews.

The training vocabulary contains **1,373 terms** in the reviewed run. Vocabulary learning happens after the split, so test reviews do not influence it. Cleaning now handles the dataset length instead of assuming exactly 1,000 rows.

## Verified result

**Test accuracy: 73%** on 200 reviews.

```text
Confusion matrix (rows = actual, columns = predicted)
             Negative  Positive
Negative        55        42
Positive        12        91
```

This is a simple baseline. It often marks negative reviews as positive and does not understand context like a language model. It is not a generative-AI project.

## Run it

Open the notebook in Jupyter or Google Colab. Keep its CSV/TSV in the notebook's working folder; opening a notebook from GitHub in Colab does not automatically upload the data.

For a local setup, clone this repository, create and activate a virtual environment, then run:

```bash
python -m pip install numpy pandas matplotlib scikit-learn nltk notebook
python -m notebook
```

Run cells from top to bottom. The first run downloads the NLTK stopword list and needs internet access.

[Verification notes](VALIDATION.md) · Tested versions: `requirements.txt`.
