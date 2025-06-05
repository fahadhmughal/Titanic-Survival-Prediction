import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading data

titanic_data = pd.read_csv(r'D:\Workspace\VS Code\Projects\Data\train.csv')
print(titanic_data.head())
print(titanic_data.shape) # getting number of records as rows
print(titanic_data.isnull().sum()) # getting number of empty columns

# Checking missing data and Handling it
# droping cabin column

titanic_data = titanic_data.drop(columns= 'Cabin' ,  axis= 1)

# replacing the missing data in 'Age' with mean value

titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

# replacing the missing data in 'Embarked' column with mode value

titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# Data Analysis
# Getting stats of data

print(titanic_data.describe())

# Getting number of survivials

print(titanic_data['Survived'].value_counts())

# Data Visulization
# Making count plot for 'Survived' column

sns.set()
sns.countplot(x = 'Survived' , data=titanic_data)
plt.show()

sns.set()
sns.countplot(x = 'Sex' , data=titanic_data)
plt.show()

# Number of survivors based on gender

sns.set()
sns.countplot(x = 'Sex' , hue= 'Survived' , data= titanic_data)
plt.show()

# Encoding the categorical column

titanic_data = titanic_data.replace({ 'Sex': {'male': 0, 'female':1} , 'Embarked': {'S':0, 'C':1, 'Q':2}}).infer_objects(copy= False)

# Seperating features and target

X = titanic_data.drop(columns= ['PassengerId' , 'Survived' , 'Name' , 'Ticket'] , axis= 1)
Y = titanic_data['Survived']

# Training the data for Training data and Testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size= 0.2 , random_state= 2)

# Model Training
# Logistic Regression Model

model = LogisticRegression(max_iter= 1000)

# Training model with train data

model.fit(X_train , Y_train)

# Model Evaluation
# Accuracy score

X_train_prediction = model.predict(X_train)

# data accuracy

X_train_accuracy_score = accuracy_score(Y_train , X_train_prediction)
print('Data accuracy of training model: ', X_train_accuracy_score)

# data accuracy of test model

X_test_prediction = model.predict(X_test)
X_test_accuracy_score = accuracy_score(Y_test , X_test_prediction)
print('Data accuracy of test model: ', X_test_accuracy_score)

# Predicting survival for a new passenger based on user input

print("\n--- Titanic Survival Prediction ---")
print("Enter passenger details:")

# Collect user input

Pclass = int(input("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd): "))
Sex = input("Sex (male/female): ").strip().lower()
Age = float(input("Age: "))
SibSp = int(input("Number of siblings/spouses aboard: "))
Parch = int(input("Number of parents/children aboard: "))
Fare = float(input("Fare paid: "))
Embarked = input("Port of Embarkation (S = Southampton, C = Cherbourg, Q = Queenstown): ").strip().upper()

# Encode categorical inputs to match training data

Sex = 0 if Sex == 'male' else 1
Embarked = {'S': 0, 'C': 1, 'Q': 2}.get(Embarked, 0)  # default to S if invalid

# Prepare input in same feature order as training

input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

# Make prediction

prediction = model.predict(input_data)

# Display result
if prediction[0] == 1:
    print("✅ The person **survived**.")
else:
    print("❌ The person **did not survive**.")


