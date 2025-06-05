# 🚢 Titanic Survival Prediction using Logistic Regression

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster. It uses the Titanic dataset and implements a **Logistic Regression** classifier in Python with scikit-learn.

---

## 📁 Dataset

The dataset contains information about Titanic passengers, including demographics, ticket details, and survival outcome.

**Key Features:**

- PassengerId  
- Pclass (Ticket class: 1st, 2nd, 3rd)  
- Name  
- Sex  
- Age  
- SibSp (Number of siblings/spouses aboard)  
- Parch (Number of parents/children aboard)  
- Ticket  
- Fare  
- Cabin  
- Embarked (Port of Embarkation)  
- Survived (Target variable: 0 = No, 1 = Yes)

**Dataset Source:**  
The Titanic dataset is publicly available from [Kaggle’s Titanic - Machine Learning from Disaster competition](https://www.kaggle.com/c/titanic/data). You can download the data from there.

---

## ⚙️ Project Workflow

1. **Load and inspect** the dataset for overview and missing values.  
2. **Handle missing data** by dropping the 'Cabin' column and filling missing 'Age' with the mean and 'Embarked' with the mode.  
3. **Visualize** key relationships using seaborn count plots (survival counts, gender distribution, survival by gender).  
4. **Encode categorical variables** (`Sex`, `Embarked`) into numeric values for modeling.  
5. **Split** the data into training and testing sets (80-20 split).  
6. **Train** a Logistic Regression model on the training data.  
7. **Evaluate** the model’s accuracy on training and test sets.  
8. **Make predictions** for new passengers based on user input features.

---

## 📊 Model Accuracy

- **Training Accuracy:** ~0.XX (actual printed by script)  
- **Test Accuracy:** ~0.XX (actual printed by script)

---
