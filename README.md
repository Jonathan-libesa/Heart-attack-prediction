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




We used four different algorithms in the model phase.





We got 87% accuracy and 88% AUC with the Logistic Regression model.




We got 83% accuracy and 85% AUC with the Decision Tree Model.




We got 83% accuracy and 89% AUC with the Support Vector Classifier Model.




And we got 90.3% accuracy and 93% AUC with the Random Forest Classifier Model.




When all these model outputs are evaluated, we prefer the model we created with the Random Forest Algorithm, which gives the best results. See you in the next project.


