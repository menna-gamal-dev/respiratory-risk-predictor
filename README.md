🫁 Respiratory Risk Predictor

An educational machine-learning project that explores whether demographic, behavioral, and saliva dielectric-response features from the Exasens dataset can be used to classify four respiratory-condition groups.

«Important: This is a research and portfolio project, not a medical diagnostic tool. The dataset is relatively small, contains substantial missingness in saliva measurements, and provides limited samples for some classes. The results should not be used for clinical decisions.»

---

🎯 Problem

Can machine-learning models distinguish between:

- COPD
- Asthma
- Respiratory infection
- Healthy Controls (HC)

using demographic, behavioral, and available saliva dielectric-response features from the Exasens dataset?

---

📊 Dataset

Source: UCI Machine Learning Repository — Exasens

- 399 records
- 4 respiratory-condition classes
- Demographic and behavioral features including:
  - Gender
  - Age
  - Smoking status
- Four saliva dielectric-response measurements:
  - Imaginary Part
  - Imaginary Part Avg
  - Real Part
  - Real Part Avg

The dataset contains substantial missingness in the saliva-response measurements. After inspecting the data, the four signal features were available for only 100 of the 399 records.

Class distribution:

Diagnosis| Records
HC| 160
Asthma| 80
Infected| 80
COPD| 79
Total| 399

Official source: UCI Machine Learning Repository
Dataset: Exasens
DOI: 10.24432/C5TC7Z

The dataset is licensed under CC BY 4.0. Please retain the appropriate attribution when redistributing or adapting the dataset.

---

🧠 Project Pipeline

Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Remove Metadata / Empty Columns
     ↓
Identify Missing Values
     ↓
Convert Signal Features to Numeric
     ↓
Exploratory Data Analysis
     ↓
Train / Test Split
     ↓
Feature Scaling
     ↓
Logistic Regression
     ↓
Random Forest
     ↓
Signal-Only Analysis
     ↓
Model Comparison
     ↓
Feature Importance Analysis
     ↓
Best Model Selection
     ↓
Save Model + Scaler
     ↓
Example Prediction

---

🧹 Data Cleaning & Preparation

The dataset was inspected before modeling to understand its structure, missing values, data types, and class distribution.

Cleaning steps

1. Metadata rows were identified and removed.
   The first two rows contained missing patient information and metadata such as "Min" and "Avg." rather than patient observations.

2. Completely empty columns were removed.

3. Metadata columns were renamed to make the signal features easier to interpret:
   
   - "Unnamed: 3" → "Imaginary Part Avg"
   - "Unnamed: 5" → "Real Part Avg"

4. ID was excluded from modeling because it is an identifier rather than a meaningful predictive feature.

5. Signal features were converted from string to numeric values while preserving their missing values.

6. Missingness in the saliva-response measurements was analyzed rather than blindly deleting all incomplete records.

7. The demographic/behavioral modeling experiment used:
   
   - Gender
   - Age
   - Smoking

8. The saliva-response features were analyzed separately because they were available for only 100 records, making direct integration into the full 399-record model inappropriate without a carefully justified missing-data strategy.

---

🔬 Exploratory Data Analysis

Several analyses were performed to understand the dataset before modeling.

Signal distributions

Boxplots were used to compare the "Imaginary Part" and "Real Part" across the four diagnosis groups.

The distributions showed considerable overlap between classes, suggesting that individual signal measurements do not provide a simple separation between respiratory-condition groups.

Feature correlation

A correlation heatmap was used to examine relationships between the four signal features.

The analysis showed strong correlations between:

- Imaginary Part and Imaginary Part Avg
- Real Part and Real Part Avg

This indicates that some signal features contain highly related information.

---

🤖 Models

Two main models were evaluated using the demographic and behavioral features:

1. Logistic Regression

Features:

Gender
Age
Smoking

"StandardScaler" was applied before training the Logistic Regression model.

2. Random Forest

The same three features were used without feature scaling, since tree-based models do not require standardization.

A separate Logistic Regression experiment was also performed using the four available saliva-response features on the subset of 100 records containing complete signal measurements.

---

📈 Results

Model Comparison

Model| Dataset| Accuracy| Weighted F1
Logistic Regression| Demographic + Behavioral| 57.50%| 49.58%
Random Forest| Demographic + Behavioral| 38.75%| 38.35%
Signal Logistic Regression| Signal-only subset| 30.00%| 31.88%

The Logistic Regression model using Gender, Age, and Smoking achieved the best performance among the experiments performed.

However, the result should be interpreted cautiously and does not demonstrate clinical diagnostic capability.

---

📋 Logistic Regression — Per-Class Performance

Class| Precision| Recall| F1-Score
Asthma| 0.45| 0.31| 0.37
COPD| 0.68| 0.81| 0.74
HC| 0.56| 0.88| 0.68
Infected| 0.00| 0.00| 0.00

The model performed relatively better on COPD and HC, while it failed to correctly identify the "Infected" class in the test set.

This highlights why overall accuracy alone is not sufficient when evaluating a multiclass classification problem.

---

🌲 Random Forest Feature Importance

The Random Forest model provided the following feature-importance ranking:

Feature| Importance
Age| 0.839
Smoking| 0.102
Gender| 0.059

Age was the most influential feature in the Random Forest model, followed by Smoking and Gender.

«Feature importance indicates how much the model relied on a feature for its predictions. It does not mean that the feature causes the respiratory condition.»

---

🔍 Key Findings

Several important observations emerged from the experiments:

- Demographic and behavioral features produced stronger performance than the signal-only experiment under the tested setup.
- "Age" was the dominant feature in the Random Forest model.
- The saliva-response features showed strong correlations with one another.
- The saliva-response measurements were available for only 100 of 399 records.
- The "Infected" class was particularly difficult for the Logistic Regression model to identify.
- The relatively small dataset and incomplete signal measurements limit the reliability and generalizability of the results.

---

⚠️ Scientific Limitations

This project has several important limitations.

Small Dataset

The dataset contains only 399 records, which limits the statistical reliability and generalizability of the models.

Missing Signal Measurements

The four saliva dielectric-response features are available for only 100 records. Simply removing the remaining records would substantially reduce the dataset and change its class composition.

Class Imbalance

The classes are not equally represented:

HC        160
Asthma     80
Infected   80
COPD       79

Therefore, accuracy should not be interpreted on its own.

Limited Generalization

Good performance on this dataset would not demonstrate that the model generalizes to other hospitals, populations, sensors, or data-acquisition protocols.

Potential Demographic Influence

The strong contribution of Age and Smoking suggests that model predictions may be substantially influenced by demographic and behavioral patterns in the dataset. Therefore, model performance should not automatically be interpreted as evidence that saliva-response measurements independently diagnose respiratory disease.

---

💾 Model Output

The best-performing model from the experiments was saved as:

respiratory_risk_logistic_model.pkl

The corresponding scaler was saved as:

respiratory_risk_scaler.pkl

An example prediction was also tested successfully through the trained pipeline.

---

🚀 Run the Project

1. Create a virtual environment

python -m venv .venv

Windows

.venv\Scripts\activate

macOS/Linux

source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Explore the analysis

Open:

notebooks/respiratory_risk_analysis.ipynb

4. Run the analysis

Execute the notebook cells sequentially to reproduce the data-cleaning, exploratory analysis, model training, evaluation, and prediction steps.

---

📁 Repository Structure

respiratory-risk-predictor/
│
├── data/
│   └── Exasens.csv
│
├── notebooks/
│   └── respiratory_risk_analysis.ipynb
│
├── results/
│
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt

---

📚 Citation

Soltani Zarrin, P., Roeckendorf, N., & collaborators.
Exasens [Dataset]. UCI Machine Learning Repository, 2020.

Please use the official UCI citation and DOI when submitting, redistributing, or publishing this project.

---

👩‍💻 Portfolio Note

This project demonstrates an end-to-end machine-learning workflow:

Real-world Dataset
        ↓
Data Quality Investigation
        ↓
Cleaning & Preparation
        ↓
Exploratory Data Analysis
        ↓
Feature Analysis
        ↓
Model Training
        ↓
Model Comparison
        ↓
Evaluation
        ↓
Model Interpretation
        ↓
Prediction

Rather than focusing only on achieving a high accuracy score, the project emphasizes data quality, reproducibility, model comparison, and responsible interpretation of machine-learning results in a healthcare-related context.