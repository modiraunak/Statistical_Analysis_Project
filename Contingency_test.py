from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

#Load Dataset
#url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
#df = pd.read_csv(url)
df = pd.read_csv(r"C:\Users\HP\Downloads\iris.csv")

contigency_table = pd.crosstab(df['species'], df['sepal_length'])
print(contigency_table)
chi2, p_value, dof, expected = chi2_contingency(contigency_table)
print(f"Chi-square statistic: {chi2}, P-value: {p_value}")

# Interpret Results
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant association between species and sepal length.")
else:
    print("Fail to reject the null hypothesis: No significant association between species and sepal length.")
    