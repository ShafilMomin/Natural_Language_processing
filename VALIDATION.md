# Verification notes

Review date: 17 September 2026.

Checks ran locally in Python 3.10. Notebook cells were executed in order with plots rendered using a non-interactive backend. Results are from this dataset and setup, not guaranteed real-world performance.

## natural_language_processing.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.73,
  "confusion_matrix": [
    [
      55,
      42
    ],
    [
      12,
      91
    ]
  ],
  "training_shape": [
    800,
    1373
  ]
}
```

## Packages

```json
{
  "numpy": "2.2.6",
  "pandas": "2.3.3",
  "matplotlib": "3.10.9",
  "scikit-learn": "1.7.2",
  "scipy": "1.15.3",
  "nltk": "3.10.3",
  "xgboost": "3.2.0"
}
```

TensorFlow was not installed in the review environment. ANN checks cover preprocessing only; CNN checks cover syntax only. Their historical training outputs are described separately in their READMEs.
