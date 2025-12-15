import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load Dataset
#url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
#df = pd.read_csv(url)
df = pd.read_csv(r"C:\Users\HP\Downloads\iris.csv")
#inspect data 
print(df.info())
print(df.describe())

del df['species']  # Remove non-numeric column for correlation

# Visualize Distribution
sns.histplot(data=df, x="sepal_length", kde=True)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()
#corelation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


