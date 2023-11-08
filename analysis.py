import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Read the dataset into a pandas DataFrame
df = pd.read_csv( r"C:\Users\cash\Documents\SEMSTER 4.1 NOTES\heart.csv")

print(df)

#Changing some column names for easy readability
new_columns = ["age","sex","cp","trtbps","chol","fbs","rest_ecg","thalach","exang","oldpeak","slope","ca","thal","target"]
df.columns = new_columns

df.head()

#SHOW INFORMATION FOR INFO
df.info()


df.describe()

#df.isnull().sum()


#Average Cholesterol Level by Chest Pain Type

plt.figure(figsize=(10, 6))
sns.barplot(x='cp', y='chol', data=df, ci='sd', palette='pastel')
plt.xlabel('Chest Pain Type')
plt.ylabel('Average Cholesterol Level')
plt.title('Average Cholesterol Level by Chest Pain Type')
plt.show()

# AGE DISTRIBUTION

plt.hist(df['age'], bins=20, color='lightblue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

 #Explore the distribution of Heart Disease by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=df, hue='target')
plt.title('Distribution of Heart Disease by Gender')
plt.show()

#Cholesterol Level by Sex
sns.boxplot(x='sex', y='chol', data=df)
plt.xlabel('Sex')
plt.ylabel('Cholesterol Level')
plt.title('Cholesterol Level by Sex')
plt.show()

#explore on  distrubtion of chest pain type
sns.countplot(x='cp', data=df)
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.title('Distribution of Chest Pain Type')
plt.show()

#Distribution of Fasting Blood Sugar
plt.figure(figsize=(7, 7))
labels = ['Fasting Sugar < 120 mg/dl', 'Fasting Sugar >= 120 mg/dl']
sizes = df['fbs'].value_counts()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['red', 'blue'])
plt.title('Distribution of Fasting Blood Sugar')
plt.show()

#Distribution of Chest Pain Type
plt.figure(figsize=(6, 6))
labels = ['sample 0', 'sample 1', 'sample 2', 'sample 3']
sizes = df['cp'].value_counts()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'blue', 'green', 'yellow'])
plt.title('Distribution of Chest Pain Type')
plt.show()

 #Explore the distribution of numerical features
numeric_features = ['age', 'trtbps', 'chol', 'thalach', 'oldpeak']
df[numeric_features].hist(bins=15, figsize=(12, 6))
plt.show()


# Explore the distribution of categorical features
plt.figure(figsize=(12, 6))
sns.countplot(x='sex', data=df, hue='target')
plt.title('Distribution of Heart Disease by Gender')
plt.show()



# Model Preperation:
df_copy = df.copy()
df_copy.head()


X = df_copy.drop(["target"],axis=1)
Y = df_copy[["target"]]

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=3)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))


#DecisionTreeClassifier

Tree_model=DecisionTreeClassifier(max_depth=10)

# fit model
Tree_model.fit(X_train,y_train)
y_pred_T =Tree_model.predict(X_test)

# Score X and Y - test and train
print("Score the X-train with Y-train is : ", Tree_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", Tree_model.score(X_test,y_test))
print("Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred_T))




# using the model K Neighbors Classifier

K_model = KNeighborsClassifier(n_neighbors = 11)
K_model.fit(X_train, y_train)

y_pred_k = K_model.predict(X_test)

print("Score the X-train with Y-train is : ", K_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", K_model.score(X_test,y_test))
print("Model Evaluation K Neighbors Classifier : accuracy score " , accuracy_score(y_test,y_pred_k))




# using the model Random Forest Classifier

RF_model = RandomForestClassifier(n_estimators = 300)
RF_model.fit(X_train, y_train)

y_pred_r = RF_model.predict(X_test)

print("Score the X-train with Y-train is : ", RF_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", RF_model.score(X_test,y_test))
print("Model Evaluation Random Forest Classifier : accuracy score " , accuracy_score(y_test,y_pred_r))

# ANALYSIS CLASSIFICATION USING RANDOMFORESTCLASSIFIER
random_forest = RandomForestClassifier(random_state=5)
random_forest.fit(X_train,y_train)
y_pred_4 = random_forest.predict(X_test)

print("The test accuracy score using Random Forest Algorithm is:",accuracy_score(y_test,y_pred_4))



RocCurveDisplay.from_estimator(random_forest, X_test, y_test, name="Random Forest Algorithm")
plt.title("Random Forest Algorithm Roc Curve and AUC")
plt.plot([0,1],[0,1],"r--")
plt.show()



parameters = {"n_estimators":[50,100,150,200],
             "criterion":["gini","entropy"],
             "max_features":["auto","sqrt","log2"],
             "bootstrap":[True,False]}


random_forest_grid = GridSearchCV(random_forest_new,param_grid=parameters)
random_forest_grid.fit(X_train,y_train)



print("Best Parameters:",random_forest_grid.best_params_)




random_forest_new2 = RandomForestClassifier(bootstrap=True,criterion="entropy",max_features="auto",n_estimators=200, random_state=5)
random_forest_new2.fit(X_train,y_train)


y_pred_4= random_forest_new2.predict(X_test)
print("accuracy score of Random Forest after hyper-parameter tuning is:",accuracy_score(y_test,y_pred_4))



RocCurveDisplay.from_estimator(random_forest_new2, X_test, y_test, name="Random Forest Algorithm")
plt.title("Random Forest Algorithm Roc Curve and AUC")
plt.plot([0,1],[0,1],"r--")