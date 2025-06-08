importpandasaspd
import numpy as np
importmatplotlib.pyplotasplt
import seaborn as sns
fromscipyimport stats
data=pd.read_csv('your_dataset.csv')
print(data.head())
column=‘sal'
data[column]=pd.to_numeric(data[column],errors='coerce')
data_clean = data[column].dropna()
mean_value = data_clean.mean()
median_value=data_clean.median()
mode_value = data_clean.mode()[0]
std_dev = data_clean.std()
variance_value = data_clean.var()
range_value=data_clean.max()-data_clean.min()
print(f"\nMean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")
print(f"StandardDeviation:{std_dev}")
print(f"Variance: {variance_value}")
print(f"Range: {range_value}")
plt.figure(figsize=(10, 6))
plt.hist(data_clean,bins=30,color='skyblue',edgecolor='black')
plt.show()
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_clean,color='lightgreen')
plt.show()
Q1=data_clean.quantile(0.25) Q3
= data_clean.quantile(0.75)
IQR=Q3-Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound=Q3+1.5*IQR
outliers=data_clean[(data_clean<lower_bound)|(data_clean>upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")
print(outliers)
column1='age’
category_counts=data[categorical_column].value_counts()
print(f"\nFrequency of each category in {column1}:")
print(category_counts)
plt.figure(figsize=(6, 3))
category_counts.plot(kind='pie',color='lightblue')
plt.show()