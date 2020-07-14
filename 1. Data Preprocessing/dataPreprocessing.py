#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
datasets = pd.read_csv('Data.csv')

#splitting independant and analyzed data
X = datasets.iloc[:, :-1].values
y = datasets.iloc[:, 3].values

#missing data (SimpleImputer is a class)
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean')
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])

# Encoding categorical data(which is not a number)
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[: , 0])
onehotencoder = OneHotEncoder(categorical_features = [0] )
X = onehotencoder.fit_transform(X).toarray()
#encoding dependant variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting datasets into train and test sets
from sklearn.model_selection import train_test_split 
X_train,X_test, y_train , y_test = train_test_split(X, y , test_size = 0.2 , random_state = 0)

""""#feature scaling
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)""""













