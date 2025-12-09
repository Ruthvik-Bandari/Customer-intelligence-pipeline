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

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| Linear Regression | Predict Total Spend | R² Score | **0.7607** |
| K-Means (K=3) | Customer Segmentation | Segments | 3 distinct groups |
| KNN Classification | Predict Gender | Accuracy | 0.4550 |

### Customer Segments Identified

| Segment | Profile | Avg Income | Avg Spend | Count |
|---------|---------|------------|-----------|-------|
| 0 | Premium Customers | $108,232 | $715 | 263 |
| 1 | Budget Buyers | $31,865 | $233 | 401 |
| 2 | Loyal Spenders | $37,616 | $706 | 336 |

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
├── AAI5025_Final_Project_Report.pdf        # Final analysis report
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or Google Colab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-intelligence-pipeline.git
   cd customer-intelligence-pipeline
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
4. **Gender-Neutral Purchasing**: Spending patterns don't significantly differ by gender

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ruthvik Bandari**
- GitHub: [@Ruthvik-Bandari](https://github.com/Ruthvik-Bandari)
- LinkedIn: [Ruthvik Bandari](https://www.linkedin.com/in/ruthvik-nath-bandari-908b00247/)

## 🙏 Acknowledgments

- Northeastern University - AAI5025 Course
- Course instructors for providing the project framework
- Scikit-learn and TextBlob documentation

---

⭐ If you found this project helpful, please consider giving it a star!
