from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Load Dataset
#url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
#df = pd.read_csv(url)
df = pd.read_csv(r"C:\Users\HP\Downloads\iris.csv")

# Inspect data
x = df['sepal_length'].values.reshape(-1, 1)
y = df['petal_length'].values
print(df.info())
print(df.describe())
print(df['species'].value_counts())
# T-test between two species
species_setosa = df[df['species'] == 'setosa']['sepal_length']
species_versicolor = df[df['species'] == 'versicolor']['sepal_length']
t_stat, p_value = ttest_ind(species_setosa, species_versicolor)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
# Linear Regression
model = LinearRegression()
model.fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(x, y)
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_squared}")
# Predict petal length for a given sepal length
sepal_length_input = np.array([[5.0]])
predicted_petal_length = model.predict(sepal_length_input)
print(f"Predicted petal length for sepal length {sepal_length_input[0][0]} cm: {predicted_petal_length[0]} cm")
#plot regression line
sns.scatterplot(x=df['sepal_length'], y=df['petal_length'], hue=df['species'])
plt.plot(df['sepal_length'], model.predict(x), color='red')
plt.title("Sepal Length vs Petal Length with Regression Line")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()