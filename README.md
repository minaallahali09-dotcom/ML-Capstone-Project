# ML-Capstone-Project

End-to-end Machine Learning capstone project on the **California Housing** dataset, covering the full pipeline: data loading, cleaning, preprocessing, EDA, modeling with three algorithm types, evaluation, and reporting.

**Course:** AIN3101 - Machine Learning, Badr University in Assiut (BUA)

## Links

- **Google Colab notebook:** https://colab.research.google.com/drive/1JlHARKEd8hPZccLVncyfWvvOrR-gi2Mo?usp=sharing
- **Final report:** [project_report.pdf](project_report.pdf) (includes the Colab link)

## Project Goal

Use the California Housing dataset to solve three tasks:

| Task | Algorithm | Goal |
|---|---|---|
| Regression | Linear Regression | Predict `median_house_value` |
| Classification | Decision Tree | Classify districts as low-price (0) or high-price (1) |
| Clustering | K-Means | Group similar districts without labels |

## Dataset

- **Source:** Kaggle - California Housing Prices (`data/housing.csv`)
- **Size:** 20,640 rows, 10 columns
- **Features:** longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity (categorical)
- **Target (regression):** `median_house_value`
- **Target (classification):** `high_price` = 1 if `median_house_value` is above the median (179,700 USD), otherwise 0

## Preprocessing

1. Missing values: 207 missing values in `total_bedrooms`, filled with the median.
2. Encoding: One-Hot Encoding for `ocean_proximity` (dataset becomes 20,640 x 14).
3. Scaling: `StandardScaler` (fitted on the training set only for supervised models).
4. Split: 80% train / 20% test, `random_state=42`.
5. For classification, `median_house_value` is removed from the features to avoid target leakage.

## Results

| Task | Model | Results |
|---|---|---|
| Regression | Linear Regression | MSE = 4,908,476,721, RMSE = 70,060 USD, R2 = 0.625 |
| Classification | Decision Tree (max_depth=5) | Accuracy = 0.815, Precision = 0.808, Recall = 0.823, F1 = 0.815 |
| Clustering | K-Means (K=5) | Silhouette Score = 0.361 |
| Extra | Random Forest Regressor | R2 = 0.817, RMSE = 48,978 USD |

**Confusion matrix (Decision Tree):** TN = 1,676, FP = 401, FN = 364, TP = 1,687.

### Key findings

- `median_income` is the strongest predictor of price (correlation 0.69).
- Prices are capped at 500,001 USD (965 districts), which limits model accuracy on expensive districts.
- Random Forest improves a lot over Linear Regression, showing that non-linear relationships matter.
- The K-Means clusters follow geography: inland districts are the cheapest, near-bay and near-ocean districts are the most expensive.

### Visualizations

**Elbow Method**

<img src="results/download%20(7).png" width="420" alt="Elbow Method">

**Final Clusters (K=5)**

<img src="results/download%20(6).png" width="420" alt="Final Clusters">

**Confusion Matrix**

<img src="results/download%20(8).png" width="360" alt="Confusion Matrix">

## Repository Structure

```
ML-Capstone-Project/
├── data/
│   └── housing.csv
├── notebooks/
│   └── California_Housing_ML_Project.ipynb
├── results/
│   └── (Elbow plot, Final clusters, Confusion matrix)
├── README.md
├── project_report.pdf
└── requirements.txt
```

## How to Run

### Option 1: Google Colab (recommended)

1. Open the Colab link above.
2. Click **Runtime > Run all**. The notebook loads the dataset automatically from this repository (`data/housing.csv`), so no file upload is needed.

### Option 2: Locally

```bash
git clone https://github.com/<your-username>/ML-Capstone-Project.git
cd ML-Capstone-Project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/California_Housing_ML_Project.ipynb
```

The notebook finds the dataset automatically in `data/housing.csv`.

## Requirements

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`

## Author

Menna Ali
