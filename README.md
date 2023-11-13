The "Heart Attack Analysis & Prediction Dataset" is a dataset used for heart attack classification":

Background:
The dataset is designed for the classification of heart attacks, a critical medical condition.
It is used for analytical and machine learning tasks related to predicting the likelihood of a heart attack based on various medical attributes.

Dataset Size:

The dataset contains 304 rows (samples) and 14 columns (features).

Features:

The dataset includes several features (attributes) that provide information about patients and their health conditions. These features can be used to predict the likelihood of a heart attack. Some of the key features include:
age: The age of the patient.
sex: The gender of the patient (0 = female, 1 = male).
cp: Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic).
trestbps: Resting blood pressure (in mm Hg).
chol: Serum cholesterol level (in mg/dl).
fbs: Fasting blood sugar (> 120 mg/dl) (1 = true, 0 = false).
restecg: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = probable or definite left ventricular hypertrophy).
thalach: Maximum heart rate achieved.
exang: Exercise-induced angina (1 = yes, 0 = no).





Within the scope of the project, we first made the data set ready for Exploratory Data Analysis(EDA)
We performed Exploratory Data Analysis(EDA).
We analyzed numerical and categorical variables within the scope of univariate analysis by using Distplot and Pie Chart graphics.
Within the scope of bivariate analysis, we analyzed the variables among each other using FacetGrid, Count Plot, Pair Plot, Swarm plot, Box plot, and Heatmap graphics.
We made the data set ready for the model. In this context, we struggled with missing and outlier values.
We used four different algorithms in the model phase.
We got 87% accuracy and 88% AUC with the Logistic Regression model.
We got 83% accuracy and 85% AUC with the Decision Tree Model.
We got 83% accuracy and 89% AUC with the Support Vector Classifier Model.
And we got 90.3% accuracy and 93% AUC with the Random Forest Classifier Model.
When all these model outputs are evaluated, we prefer the model we created with the Random Forest Algorithm, which gives the best results. See you in the next project.
oldpeak: ST depression induced by exercise relative to rest.
slope: Slope of the peak exercise ST segment.
ca: Number of major vessels colored by fluoroscopy.
thal: Thalassemia (a type of blood disorder) results (3 = normal, 6 = fixed defect, 7 = reversible defect).
Target Variable:
The target variable in this dataset is often output or target, which indicates the presence (1) or absence (0) of a heart attack

The target variable in this dataset is often output or target, which indicates the presence (1) or absence (0) of a heart attack
