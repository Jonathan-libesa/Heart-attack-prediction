import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset into a pandas DataFrame
df = pd.read_csv( r"C:\Users\cash\Documents\SEMSTER 4.1 NOTES\heart.csv")

print(df)



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

# Explore the distribution of Heart Disease by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=df, hue='output')
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
numeric_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
df[numeric_features].hist(bins=15, figsize=(12, 6))
plt.show()


# Explore the distribution of categorical features
plt.figure(figsize=(12, 6))
sns.countplot(x='sex', data=df, hue='output')
plt.title('Distribution of Heart Disease by Gender')
plt.show()
