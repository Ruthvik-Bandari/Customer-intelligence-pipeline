# Customer Intelligence Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive data science pipeline for analyzing customer behavior, sentiment, and segmentation using machine learning techniques. This project demonstrates end-to-end data processing from raw database extraction to predictive modeling.

## 📋 Project Overview

This project implements a complete **Customer Intelligence Pipeline** that:
- Extracts and cleans raw customer data from a SQLite database
- Engineers meaningful features using NLP and mathematical transformations
- Visualizes key business insights through exploratory data analysis
- Applies machine learning models for prediction and segmentation

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Data Acquisition** | SQL-based extraction from SQLite database |
| **Data Wrangling** | Currency string cleaning, missing value imputation, duplicate removal |
| **Feature Engineering** | Spend per Visit calculation, NLP sentiment analysis |
| **Visualization** | Income distribution, education comparisons, correlation analysis |
| **Predictive Modeling** | Linear Regression, K-Means Clustering, KNN Classification |

## 📊 Results Summary

### Model Performance

| Model | Task | Metric | Score | Verdict |
|-------|------|--------|-------|---------|
| Linear Regression | Predict total spend | R² | **0.7607** on 200 held-out rows | Works |
| K-Means (K=3) | Segmentation | Segment sizes | 263 / 401 / 336 | Works |
| KNN (k=5) | Predict gender | Accuracy | 0.4550 | **Fails — below baseline** |

> **The KNN model does not work, and that is the finding.** Its 0.4550 accuracy sits *below* the
> majority-class baseline of **0.5090**. The notebook computes that baseline, compares against it,
> and prints "The model does not significantly outperform baseline." Read this as evidence that
> **income and spending carry no gender signal** in this dataset, which is a legitimate negative
> result, not as a model to improve.

### Dataset

1,002 source rows, 1,000 after removing 2 exact duplicates. 117 `Education` values arrived as the
literal string `'nan'` and were imputed to the mode.

### Customer Segments Identified

| Segment | Profile | Avg Income | Avg Spend | Count |
|---------|---------|------------|-----------|-------|
| 0 | Premium Customers | $108,232 | $715 | 263 |
| 1 | Budget Buyers | $31,865 | $233 | 401 |
| 2 | Loyal Spenders | $37,616 | $706 | 336 |

Centroids are inverse-transformed back to dollar space so they stay interpretable. Regression
coefficients show visit frequency dominates: **37.01 per visit against 0.0009 per income dollar**,
with total spend correlating to visits at **0.8818**.

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **SQLite3** - Database connectivity
- **TextBlob** - NLP sentiment analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models

## 📁 Project Structure

```
customer-intelligence-pipeline/
│
├── AAI5025_Final_Project_Notebook.ipynb   # Main Jupyter notebook
├── user_data.db                            # SQLite database (raw data)
├── final_processed_capstone_data.csv       # Processed output data
├── Final_Project_Report_Instructions.pdf   # Assignment brief
├── README.md                               # Project documentation
├── gitignore                               # NOTE: missing leading dot, currently inactive
└── requirements.txt                        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or Google Colab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ruthvik-Bandari/Customer-intelligence-pipeline.git
   cd Customer-intelligence-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook AAI5025_Final_Project_Notebook.ipynb
   ```

### Running in Google Colab

1. Upload `AAI5025_Final_Project_Notebook.ipynb` to Google Colab
2. Upload `user_data.db` to the Colab file system
3. Run all cells sequentially

## 📈 Visualizations

### Income Distribution
- Right-skewed distribution with Mean: $53,882 and Median: $41,000
- Long tail of high-income customers extending to $200K

### Education vs Spending
- Masters: $47.26 per visit (highest)
- High School: $44.40 per visit
- Bachelors: $43.57 per visit (lowest)

### Correlation Analysis
- Income vs Sentiment: 0.1654 (weak positive)
- Word Count vs Sentiment: 0.1189 (weak positive)

## 🔍 Key Insights

1. **Strong Spending Predictor**: Visit frequency and income explain 76% of spending variation
2. **Income Doesn't Drive Satisfaction**: Customer sentiment is independent of income level
3. **Loyal Spenders Segment**: Moderate-income customers with high spending represent a key retention target
4. **Gender-Neutral Purchasing**: Spending patterns carry no usable gender signal — the KNN
   classifier scored below its own majority-class baseline, which is the evidence for this claim

## 📝 Methodology

### Data Cleaning
```python
# Clean currency formatting using vectorized operations
df['Clean_Income'] = (
    df['Raw_Income']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

# Impute missing Education with mode
df['Education'] = df['Education'].fillna(df['Education'].mode()[0])
```

### Feature Engineering
```python
# Spend per Visit
df['Spend_per_Visit'] = df['Total_Spend'] / df['Avg_Visits']

# Sentiment Analysis
from textblob import TextBlob
df['Sentiment_Score'] = df['Review_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

### Machine Learning Pipeline
```python
# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['User_Segment'] = kmeans.fit_predict(X_scaled)
```


## ✅ Methodological notes

Practices worth pointing at, since they are the difference between a pipeline that generalises and
one that leaks:

- `StandardScaler` is **fit on the training split only** and `.transform()` applied to test.
- Splitting is **stratified** on a target whose minority class is 45 of 1,000.
- Scaling happens **before** K-Means, with centroids inverse-transformed for interpretation.
- Cleaning is fully **vectorised** — no Python row loops.
- `random_state=42` and an explicit `n_init=10` throughout, so runs reproduce.

## ⚠️ Limitations

- Single dataset of 1,000 rows, single train/test split, no cross-validation.
- Sentiment is TextBlob polarity on short review text, which is coarse.
- The gender classifier is a documented negative result, not a working model.
- `user_data.db` and `.DS_Store` are committed; `gitignore` is missing its leading dot so it has no
  effect.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ruthvik Bandari**
- GitHub: [@Ruthvik-Bandari](https://github.com/Ruthvik-Bandari)
- LinkedIn: [Ruthvik Bandari](https://www.linkedin.com/in/ruthvik-nath-bandari/)

## 🙏 Acknowledgments

- Northeastern University - AAI5025 Course
- Course instructors for providing the project framework
- Scikit-learn and TextBlob documentation

---

⭐ If you found this project helpful, please consider giving it a star!
