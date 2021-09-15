import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
df.describe()
df.info()
df['Species'].value_counts()
df.isnull().sum()
df['Sepal.Length'].hist()
df['Sepal.Width'].hist()
df['Petal.Length'].hist()
df['Petal.Width'].hist()
colors = ['red', 'orange', 'blue']
species = ['setosa','versicolor','virginica']
for i in range(3):
  x = df[df['Species'] == species[i]]
  plt.scatter(x['Sepal.Length'], x['Sepal.Width'], c = colors[i], label=species[i])
  plt.xlabel('Sepal Length')
  plt.ylabel('Sepal Width')
  plt.legend()
  for i in range(3):
  x = df[df['Species'] == species[i]]
  plt.scatter(x['Petal.Length'], x['Petal.Width'], c = colors[i], label=species[i])
  plt.xlabel('Petal Length')
  plt.ylabel('Petal Width')
  plt.legend()
  for i in range(3):
  x = df[df['Species'] == species[i]]
  plt.scatter(x['Sepal.Length'], x['Petal.Length'], c = colors[i], label=species[i])
  plt.xlabel('Sepal Length')
  plt.ylabel('Petal Length')
  plt.legend()
  for i in range(3):
  x = df[df['Species'] == species[i]]
  plt.scatter(x['Sepal.Width'], x['Petal.Width'], c = colors[i], label=species[i])
  plt.xlabel('Sepal Width')
  plt.ylabel('Petal Width')
  plt.legend()
  df.corr()
  corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head()
from sklearn.model_selection import train_test_split


X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

print("Accuracy: ",model.score(x_test, y_test) * 100)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)
 print("Accuracy: ",model.score(x_test, y_test) * 100)
 from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)
