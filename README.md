# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method
# NAME: YUVAN SUNDAR S
# REG NO: 212223040250
# CODING AND OUTPUT:
```py
# Feature Scaling
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/bbb481e4-2ab9-4d64-a763-498514c1a8fd)
```py
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/fdb338db-c05f-4ef9-b4ea-2d7eae969d5a)

```py
df.dropna()
```
![image](https://github.com/user-attachments/assets/d8047121-74df-49e9-b684-ff7387be8daf)

```py
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/7cbda600-2bd7-4383-9a14-4edeae6ad5e0)

```py
# Standard Scaling
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/user-attachments/assets/3e2975f6-cd6b-4a59-8b61-b5527763b5b9)

```py
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/a8d2b3d7-8a38-4eed-a79c-e08b8ca939de)

```py
# MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/03692cc2-7728-47b6-9ea3-1b4fcf4dcbcb)

```py
# MAXIMUM ABSOLUTE SCALING:
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```
![image](https://github.com/user-attachments/assets/f1f508a8-579a-4192-94c4-84a57f082bec)

```py
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/2ad05eb3-4802-48aa-8e5f-2ff4be933ca9)

```py
# ROBUST SCALING:
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/3ff7743c-252a-4438-b6f7-6e5944c577c6)

```py
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/d85bc4d8-c848-47b9-9f63-d5992b246d14)

```py
# FEATURE SELECTION:
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/21bf0b27-a839-4eed-8715-c62bef63e360)

```py
df
```
![image](https://github.com/user-attachments/assets/95333d08-8293-4c65-8322-213f8dc5f9cb)

```py
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/d7b92671-d5c3-4b57-b42c-398637051fa2)

```py
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/5d931072-bf25-4246-82cc-22b32765fe3e)

```py
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/04bda814-806a-49b1-9a42-621f7f2fbf4c)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/fe2865dc-72a6-4d9c-a247-b66109a603c2)

```py
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/8e3226af-6a26-4585-bcc8-a79526ce0dc6)

```py
# FILTER METHOD:
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/8261ce6d-36f6-4113-951e-592098abd23e)

```py
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/4ecaa7e5-0034-4c43-a3d7-eba601289cdd)

```py
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_chi2 = 6
 selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
 X_chi2 = selector_chi2.fit_transform(X, y)
 selected_features_chi2 = X.columns[selector_chi2.get_support()]
 print("Selected features using chi-square test:")
 print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/418b0bb4-a2d6-45e1-b738-854f739d3b18)

```py
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
       'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/6e923be6-b8f2-4578-bcd2-3287588adc4a)

```py
 y_pred = rf.predict(X_test)
 from sklearn.metrics import accuracy_score
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/a86f3bb5-aa42-45e7-9d8e-092d3a93bb7f)

```py
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/8d688839-2bab-432c-95d3-30f1bcb6bb94)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/f7c99b3a-0679-4d16-91e2-8c4e66ea9a75)

```py
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/d43b6676-3f46-4be6-b89d-296bd6352d7a)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/1d7b7115-f148-4079-b968-da4e65ef6fac)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
