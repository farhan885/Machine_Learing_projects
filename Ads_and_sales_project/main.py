#importing basic libraries
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

st.header("Advertising and Sales Project")
st.subheader('We are predicting sales of products based on Advertising')
st.sidebar.header("Advertising and Sales Project")

st.write('Data view')
df = pd.read_csv('advertising.csv')
st.write(df.head(20))

st.write('Basic statistical report')
st.write(df.describe())

st.write('Visualizing data columns using pairplot from seaborn library')
st.pyplot(sns.pairplot(df, x_vars = ['TV','Radio','Newspaper'], y_vars='Sales',size=7, kind='reg'))

def func():
    plt.figure(figsize=(12, 10))
    sns.pairplot(df)
    plt.show()
st.pyplot(func())

# Spliting data for train and test
X_train, X_test, y_train, y_test = train_test_split(df[['TV','Radio', 'Newspaper']], df['Sales'], test_size=0.2, random_state=100)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write('Mean Squared Error between y_test values and y_pred values')
mse = mean_squared_error(y_test, y_pred)
r_score = r2_score(y_test, y_pred)
st.write(mse, r_score)