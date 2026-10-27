# Diabetes Risk Prediction: Naive Bayes vs. K-Nearest Neighbors
 
Multi-class classification of diabetes risk (**Low / Moderate / High**) for 15,000 patients, comparing a Categorical Naive Bayes model against a cross-validated KNN pipeline. The comparison goes beyond overall accuracy to per-class recall and macro-F1, because the target is imbalanced.
 
## Dataset
 
[Diabetes Risk Prediction (Kaggle)](https://www.kaggle.com/datasets/mansiaggarwal88/diabetes-risk-prediction/data): 15,000 patients, 19 columns including demographics, lifestyle (diet, activity, sleep, smoking, alcohol), and clinical measures (BMI, fasting blood sugar, HbA1c, blood pressure, waist circumference). Target: `diabetes_risk`.
 
## Approach
 
1. **Cleaning:** dropped non-informative columns (`patient_id`, `city`, `income_bracket`). Smoking status (472 missing) and alcohol consumption (3,788 missing) were filled with an explicit `Unknown` category rather than dropping rows, preserving data and keeping non-response as a possible signal.
2. **EDA:** class distribution (imbalanced toward Low), descriptive statistics, joint plots, and a mixed-type association heatmap using `dython` (handles categorical and numeric features together). Fasting blood sugar and HbA1c showed the strongest association with the target.
3. **Model 1, Categorical Naive Bayes:** 70/30 split, ordinal encoding of categoricals, and decile binning of numeric features. Bin edges were learned on the training set only, to avoid leakage.
4. **Model 2, KNN:** `ColumnTransformer` + `Pipeline` (StandardScaler for numeric, OneHotEncoder for categorical). `GridSearchCV` with 5-fold CV tuned K. The first search hit the edge of the grid (K=31), so the range was widened and the best value settled at **K=43**.
5. **Evaluation:** confusion matrices, classification reports, train vs. test gap, macro-F1.
## Results
 
| Model | Test accuracy | Macro-F1 | Recall (High / Low / Moderate) |
|---|---|---|---|
| Categorical Naive Bayes | 76.9% | 0.73 | 0.74 / 0.83 / 0.63 |
| KNN (K=43) | 76.0% | 0.68 | 0.59 / 0.94 / 0.41 |
 
- Train and test accuracy were within about 2 points for both models, so neither is badly overfit.
- Overall accuracy is nearly identical, but the models behave differently. KNN is better at identifying Low risk, while Naive Bayes is considerably better on Moderate and High risk.
- **Naive Bayes was selected as the better model** for this dataset on macro-F1 and minority-class recall.
- Both models struggle with the Moderate class, which is frequently confused with its neighbors.

## Tech Stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, dython
 
## Running It
 
```bash
pip install pandas numpy scikit-learn seaborn matplotlib dython
# download diabetes_risk.csv from the Kaggle link above into the project folder
jupyter notebook diabetesrisk.ipynb
```
