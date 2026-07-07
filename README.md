# 🍽️ Restaurant Review Sentiment Analysis using NLP

A Natural Language Processing project that classifies restaurant reviews as **Positive** or **Negative** using text preprocessing, Bag of Words model, and Naive Bayes classification.

---

## 📌 Problem Statement

Online restaurant reviews are written in free-form text — messy, informal, and full of noise. This project builds an NLP pipeline that automatically reads a restaurant review and predicts whether the customer had a **positive or negative experience**.

---

## 📂 Repository Structure

```
├── natural_language_processing.ipynb   # Main notebook with full NLP pipeline
├── Restaurant_Reviews.tsv             # Dataset (1000 labeled reviews)
└── README.md
```

---

## 📊 Dataset

**File:** `Restaurant_Reviews.tsv`
**Format:** TSV (Tab Separated Values)
**Size:** 1000 restaurant reviews

| Column | Description |
|---|---|
| **Review** | Raw text of the customer review |
| **Liked** | Target → 1 = Positive, 0 = Negative |

**Sample Reviews:**
```
"Wow... Loved this place."          → 1 (Positive) ✅
"Crust is not good."               → 0 (Negative) ❌
"Not tasty and the texture was nasty." → 0 (Negative) ❌
```

---

## ⚙️ Project Pipeline

### Part 1 — Text Cleaning (Preprocessing)

Raw text is noisy — numbers, punctuation, capital letters, and common words all add noise. Cleaning steps:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    # Step 1: Remove everything except letters
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # Step 2: Lowercase everything
    review = review.lower()
    
    # Step 3: Split into words
    review = review.split()
    
    # Step 4: Stemming + Remove stopwords
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')  # Keep 'not' — it changes meaning!
    all_stopwords.remove('is')   # Keep 'is' — important for context
    review = [ps.stem(word) for word in review 
              if word not in set(all_stopwords)]
    
    review = ' '.join(review)
    corpus.append(review)
```

**What each step does:**

| Step | Before | After |
|---|---|---|
| Remove non-letters | "Wow... Loved this place!" | "Wow  Loved this place " |
| Lowercase | "Loved" | "loved" |
| Remove stopwords | "loved this place" | "loved place" |
| Stemming | "loving", "loved" | "love", "love" |

> ⚠️ **Important:** `'not'` was intentionally kept in stopwords — removing it would flip the meaning of negative reviews like "not good" → "good"!

---

### Part 2 — Bag of Words Model

Convert cleaned text into numbers that a model can understand:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
```

**How Bag of Words works:**
```
Vocabulary: ['amaz', 'bad', 'delici', 'good', 'love', 'nasti', 'not', ...]
                                                                    
Review 1: "wow love place"     → [0, 0, 0, 0, 1, 0, 0, ...]
Review 2: "crust is not good"  → [0, 0, 0, 1, 0, 0, 1, ...]

Each review = a vector of word counts (1500 features!)
```

---

### Part 3 — Train-Test Split

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
# 800 reviews for training, 200 for testing
```

---

### Part 4 — Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
```

**Why Naive Bayes for NLP?**
- Works excellently with high-dimensional text data
- Fast to train — even with 1500 features
- Strong baseline for sentiment analysis tasks

---

### Part 5 — Evaluation

```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

---

## 📈 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **73%** |
| True Negatives (Correctly predicted Negative) | 55 |
| False Positives (Negative predicted as Positive) | 42 |
| False Negatives (Positive predicted as Negative) | 12 |
| True Positives (Correctly predicted Positive) | 91 |

**Confusion Matrix:**
```
[[55  42]
 [12  91]]
```

---

## 🧠 Key Concepts Used

- **Text Preprocessing** — Removing noise, lowercasing, punctuation removal
- **Stopword Removal** — Filtering common words that don't carry meaning
- **Porter Stemmer** — Reducing words to their root form (loving → love)
- **Bag of Words** — Converting text to numerical feature vectors
- **CountVectorizer** — Building vocabulary and encoding reviews
- **Gaussian Naive Bayes** — Probabilistic classifier using Bayes theorem
- **Confusion Matrix** — Evaluating true/false positives and negatives

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Loading TSV dataset |
| `numpy` | Array operations |
| `re` | Regular expressions for text cleaning |
| `nltk` | Stopwords and Porter Stemmer |
| `scikit-learn` | CountVectorizer, GaussianNB, evaluation metrics |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install numpy pandas scikit-learn nltk
```

**3. Download NLTK data**
```python
import nltk
nltk.download('stopwords')
```

**4. Run the notebook**
Open `natural_language_processing.ipynb` in Jupyter Notebook or Google Colab and run all cells.

---

## 👤 Author

**Shafil** — AI & ML Engineering Student
B.Tech Final Year | Specialization: Computer Vision & Deep Learning

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
