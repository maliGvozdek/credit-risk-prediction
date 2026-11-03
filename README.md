# Credit Risk Prediction – Knowledge Discovery in Data

Team project for the course Knowledge Discovery in Data at FOI, University of Zagreb. A machine learning model that predicts whether a bank client is a good or bad credit risk.

## Dataset

- **Source:** Given by the professor
- **Size:** 1,000 clients, 10 attributes
- **Attributes:** age, sex, job, housing, saving accounts, checking account,
  credit amount, duration, purpose, and the target variable **Risk** (good / bad)
- Two columns have missing values: saving accounts (817 of 1,000 filled)
  and checking account (606 of 1,000 filled)

## Approach

- Split the data into training and test sets (80:20)
- Trained a **Random Forest classifier** (scikit-learn)
- The data is imbalanced (about 70% good and 30% bad clients), so we used class
  weights (`class_weight={0: 2.33, 1: 1}`) to reduce the model's bias towards
  the majority class
- Raised the classification threshold from 0.5 to 0.8, so the model only labels
  a client as "good" when it is confident
- Analysed feature importance and selected the 18 features that gave the best results

## Results

| Model | Accuracy | Recall (bad risk) | Recall (good risk) | Precision (good risk) | F1 (good risk) |
|---|---|---|---|---|---|
| Baseline | 58% | 87% | 46% | 89% | 0.61 |
| Optimised | 60% | 82% | 51% | 87% | 0.64 |

The model is deliberately cautious. It catches most risky clients (82–87%), but it also labels many good clients as risky. When it does label a client as good, it is right about 87% of the time. For a bank, missing a risky client usually costs more than rejecting a good one, so this trade-off was intentional.

Feature selection and threshold tuning improved accuracy from 58% to 60% and recall for good clients from 46% to 51%.

## Tools

Python · pandas · scikit-learn · Google Colab

## Team & my contribution

**Team project:** Luka Gvozdanović and Dominik Katunić.  
**My part:** data cleaning, model training, evaluation, documentation
