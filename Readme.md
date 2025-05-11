# Diabetes Prediction Using Machine Learning

## Project Overview
This project develops a machine learning model to predict diabetes in patients based on diagnostic measurements. Using the PIMA Indians Diabetes Dataset, we build and compare several classification algorithms to identify the most effective approach for early diabetes detection.


## Objectives
- Analyze factors that influence diabetes diagnosis
- Develop accurate prediction models for early diabetes detection
- Compare performance of different machine learning algorithms
- Identify the most important features for prediction
- Create a robust model that could potentially assist healthcare professionals

## Dataset
The dataset includes diagnostic measurements for 768 patients, with the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration (mg/dL)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function (genetic influence)
- Age: Age in years
- Outcome: Target variable (0 = No diabetes, 1 = Diabetes)

## Methodology

### 1. Exploratory Data Analysis
- Performed statistical analysis of the dataset to understand distributions and relationships
- Visualized feature distributions using histograms and box plots
- Created correlation matrices to identify relationships between features
- Analyzed feature distributions in relation to the target variable
- Identified class imbalance in the dataset (65% non-diabetic, 35% diabetic)

### 2. Data Preprocessing
- Handled missing values using median imputation based on outcome class
- Applied target-based imputation to maintain the statistical relationship between features and target
- Detected and handled outliers using multiple techniques:
  - Interquartile Range (IQR) method
  - Local Outlier Factor (LOF) for multivariate outlier detection
  - Visual inspection using box plots for each feature
- Replaced extreme outliers with boundary values to limit their influence

### 3. Feature Engineering
- Created new categorical features for improved model performance:
  - BMI categories (Underweight, Normal, Overweight, Obesity 1/2/3)
  - Insulin score (Normal/Abnormal based on clinical thresholds)
  - Glucose categories (Low, Normal, Overweight, Secret)
- Applied feature scaling using StandardScaler and RobustScaler
- Combined original features with derived categorical features

### 4. Model Development
We implemented and evaluated several machine learning algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

### 5. Hyperparameter Tuning
- Applied GridSearchCV with cross-validation to optimize each model
- Tuned parameters for each algorithm, including:
  - Learning rates
  - Tree depths
  - Regularization parameters
  - Ensemble sizes
  - Split criteria

### 6. Model Evaluation
- Used a train-test split (80% train, 20% test) for evaluation
- Implemented metrics for performance assessment:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Compared algorithms using ROC curves and bar charts
- Analyzed confusion matrices to understand error patterns

## Results
- **XGBoost** and **Gradient Boosting** emerged as the top-performing models
- Achieved accuracy rates of approximately **85%** on the test set
- ROC-AUC scores reached **0.89** for the best models
- Key features driving predictions:
  - Glucose level (most important predictor)
  - Age
  - BMI
  - DiabetesPedigreeFunction

![Performace_Diabetes_Prediction_Models](https://github.com/user-attachments/assets/d7962f83-e78b-4f36-a5b7-790bb53071eb)
![ROC_Curves_Diabetes_Prediction_Models](https://github.com/user-attachments/assets/01282188-9027-4107-8b88-0431a38eb1ab)


## Conclusion
This project demonstrates the effectiveness of ensemble methods (particularly XGBoost and Gradient Boosting) for diabetes prediction. The models developed could potentially serve as valuable screening tools in healthcare settings, helping identify patients at risk of diabetes for further clinical evaluation.

## Future Work
- Incorporate additional clinical features if available
- Explore deep learning approaches
- Develop a web application for healthcare providers
- Investigate feature importance across different demographic groups
- Apply model explainability techniques for better interpretability

## Tools and Technologies
- Python
- Pandas, NumPy for data manipulation
- Scikit-learn for modeling and evaluation
- Matplotlib, Seaborn for visualization
- Jupyter Notebook for development
