# E-commerce Customer Purchase Prediction

A machine learning project that estimates whether an e-commerce customer is likely to make a purchase based on customer attributes and browsing-session behavior. The project includes data analysis, feature engineering, model evaluation, explainability, and an interactive Streamlit application.

## Project Overview

Online retailers want to understand customer purchase intent so they can improve customer engagement and marketing decisions. This project builds a classification pipeline to estimate purchase probability from customer and session information.

**Project type:** Machine Learning — Binary Classification  
**Frontend:** Streamlit  
**Best-performing model by test F1-score:** Tuned Random Forest  
**Dataset:** Synthetic data created for this project

## Features

- Data understanding and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Logistic Regression, Decision Tree, and Random Forest comparison
- Cross-validation and hyperparameter tuning
- Model evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix
- Feature importance and permutation importance
- Purchase probability prediction
- Interactive Streamlit dashboard

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook / Google Colab

## Feature Engineering

The project creates four additional features:

| Feature | Description |
|---|---|
| `Pages_Per_Minute` | Pages viewed divided by session duration |
| `Products_Per_Page` | Products viewed divided by pages viewed |
| `Avg_Spending_Per_Previous_Purchase` | Historical spending divided by previous purchases |
| `Engagement_Level` | Low, Medium, or High engagement category |

`Customer_ID` is excluded from the model inputs because it is an identifier rather than a behavioral feature.

## Model Evaluation

The models were evaluated on a stratified train/test split. The following are the recorded test-set results.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7728 | 0.8184 | 0.8760 | 0.8462 | 0.8180 |
| Decision Tree | 0.7673 | 0.8077 | 0.8844 | 0.8443 | 0.8111 |
| Random Forest (baseline) | 0.7685 | 0.8116 | 0.8799 | 0.8444 | 0.8122 |
| Random Forest (tuned) | 0.7700 | 0.8084 | 0.8883 | 0.8465 | 0.8152 |

The tuned Random Forest achieved the highest test F1-score among these models. Logistic Regression achieved the highest test ROC-AUC among the compared models.

## Explainability and Business Insights

The analysis identified cart activity, customer engagement, products viewed, pages viewed, and session duration as useful predictive signals.

These findings describe associations in the dataset. They do not establish that any feature causes a purchase.

## Streamlit Application

The dashboard allows users to enter customer and session information and view:

- The engineered features
- Estimated purchase probability
- Predicted purchase class
- A brief interpretation of customer activity

The application uses a classification threshold of **0.35 (35%)**: probabilities at or above this threshold are classified as a predicted purchase.

**Threshold limitation:** The 0.35 threshold was selected using an analysis of test-set results. For an unbiased final evaluation, threshold selection should be performed on validation data and the final model evaluated on a separate untouched test set.

## Dataset Disclosure

The dataset used in this project is **synthetically generated** for learning and demonstration. The results should not be interpreted as validated performance on real customer transactions or as evidence of real-world business impact.

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ecommerce-customer-purchase-prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment

Use Python 3.12.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Project Structure

```text
ecommerce-customer-purchase-prediction/
├── app.py
├── model.pkl
├── ecommerce_purchase_prediction.ipynb
├── README.md
├── requirements.txt
├── submission.csv
└── ecommerce_customer_purchase_raw_synthetic.csv
```

The synthetic dataset file is optional if you only want to run the saved model and frontend. Include it if you want to make the data preparation and analysis reproducible.

## Future Improvements

- Evaluate the model on real, appropriately consented customer data.
- Select the decision threshold using validation data.
- Add automated tests for input validation and predictions.
- Add more visual analytics to the dashboard.
- Deploy the application and monitor model performance.

## Conclusion

This project demonstrates an end-to-end machine learning workflow, from data preparation and model comparison to explainability and an interactive prediction interface. It is intended as a learning and portfolio project, with synthetic data and evaluation limitations documented explicitly.