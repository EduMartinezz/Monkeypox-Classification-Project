### **Monkeypox Classification Using Machine Learning**

## **Overview**
This repository focuses on applying machine learning techniques to predict Monkeypox outcomes using real-world clinical and demographic data. It aims to build an accurate classification model while identifying the most significant factors influencing Monkeypox PCR results. The project explores various machine learning algorithms and advanced ensemble methods, making it a valuable tool for public health initiatives.


## **Project Highlights**
- **Data Cleaning**: Removed inconsistencies in variables like **Age** and categorical symptoms. Imputed missing values to create a robust dataset.
- **Predictive Modeling and Comprehensive Model Comparison**: Used multiple machine learning algorithms including Logistic Regression, **Naïve Bayes**, **Random Forest**, **XGBoost, and Gradient Boosting**.
- Compared traditional classifiers (e.g., **Logistic Regression, SVM**) with ensemble and boosting techniques (e.g., **XGBoost, Gradient Boosting**).
- **Feature Importance Analysis**: Identified key risk factors for Monkeypox, including `Age`, `HIV Infection`, and `Encoded Systemic Illness`.
- **Visualization**: Included interpretable visualizations such as SHAP values, feature importance, correlation heatmaps, and model comparison.
-  **Feature Interpretability**:SHAP and Feature Importance visualizations provided insights into model decision-making.
  
  **Advanced Techniques**:
  - Hyperparameter tuning using Grid Search and Bayesian Optimization.
  - Stacking and Voting classifiers for ensemble learning.
  - Cross-validation for generalization performance.
  - Bootstrap confidence intervals to validate model accuracy.

## **Dataset**
**Features**:
  - Age
  - Encoded Systemic Illness
  - HIV Infection
  - Symptoms such as Rectal Pain, Sore Throat, and Oral Lesions.

**Target**: `MPOX_Result` (Binary: 0 = Negative, 1 = Positive)


## **Directory Structure**

monkeypox-classification/
├── data/
│   ├── Monkeypox_Dataset.csv  # Original dataset (if permitted)
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis
│   ├── preprocessing.ipynb    # Data Preprocessing
│   ├── modeling.ipynb         # Model Building and Evaluation
├── src/
│   ├── data_cleaning.py       # Scripts for cleaning and transforming data
│   ├── feature_engineering.py # Feature engineering and scaling
│   ├── model_training.py      # Training and evaluating models
│   ├── model_inference.py     # For deploying/predicting unseen data
├── images/
│   ├── distribution_mpox_results.png
│   ├── correlation_heatmap.png
│   ├── feature_importance_rf.png
│   ├── shap_summary_plot.png
│   ├── partial_dependence_plot.png
│   ├── model_comparison.png
├── README.md
├── requirements.txt           # Dependencies for the project
├── LICENSE                    # Licensing information (optional)
├── .gitignore                 # Ignore unnecessary files (e.g., datasets, cache)



## **Getting Started**

**1. Clone the Repository**   
git clone https://github.com/yourusername/monkeypox-classification.git
cd monkeypox-classification

**Install Dependencies**   
pip install -r requirements.txt

## **3. Explore Notebooks** 
Navigate to the notebooks/ folder to review:
- eda.ipynb for exploratory data analysis.
- preprocessing.ipynb for data preprocessing steps.
- modeling.ipynb for model building and evaluation.

### 4. **Run Python Scripts**
### Data Cleaning:
python src/data_cleaning.py

## Feature Engineering:
python src/feature_engineering.py

### **Model training and evaluation:** 
python src/model_training.py

### **Visualizations** 
**Distribution of MPOX Results, Correlation Heatmap, Feature Importance - Random Forest, SHAP Summary Plot, Model Comparison, Partial Dependence Plot.**


## **Evaluation Metrics**
The following models were evaluated:

**Model**	                **Accuracy**
Gradient Boosting	          70.06%
AdaBoost	                  69.56%
Naïve Bayes	                68.22%
Logistic Regression	        68.14%
SVM	                        67.92%
XGBoost	                    67.86%
KNN	                        63.88%
Random Forest	              62.64%
Decision Tree	              58.46%


### **Advanced Techniques**
**Cross-Validation:**
- Achieved a mean CV accuracy of 69.10% with Gradient Boosting.

## Hyperparameter Tuning:
- Used GridSearchCV and Bayesian Optimization to fine-tune model parameters.

## Interpretability:
- SHAP and Partial Dependence Plots for feature importance.


### Relevance to Public Health
The project contributes to:
- Early Detection: Identifies high-risk patients based on clinical features.
- Epidemiological Insights: Highlights significant symptoms and risk factors.
- Healthcare Resource Allocation: Prioritizes patients based on predictions.


### **Future Work**   
- Enhance model accuracy using deep learning approaches.
- Integrate real-time data streams for continuous learning.
- Build a web-based application for Monkeypox risk assessment.
- Investigate time-series trends for monitoring Monkeypox outbreaks.

