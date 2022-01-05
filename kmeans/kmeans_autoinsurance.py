import pandas as pd                                                   # importing libraries
from matplotlib import pyplot as plt
from sklearn.cluster import	KMeans

df = pd.read_csv("AutoInsurance.csv")                                    # importing dataset
df

df1 = df.drop(["Customer","State","Effective To Date"] , axis = 1)       # dropping nominal column
df1
df2 = pd.get_dummies(df1 , drop_first = True)                            # dummy variables for char column
df2

# EDA part and data cleansing *******************************************************************************

df1.describe()                                                           # describing dataset
df1.info()                                                               # checking for datatypes and null values
df.duplicated().sum()                                                    # checking duplicacy
plt.boxplot(df2)                                                         # checking for outliers
# plotting scatter plots
plt.scatter(df1["Customer Lifetime Value"],df1["Monthly Premium Auto"])
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Monthly Premium Auto")
plt.scatter(df1["Customer Lifetime Value"],df1["Income"])
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Income")
plt.scatter(df1["Customer Lifetime Value"],df1["Total Claim Amount"])
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Total Claim Amount")
plt.matshow(df1.corr())                                                  # plotting matshow for correlations
# normalising to scale free and unit free
def norm1(i):
    x = (i - i.min())/(i.max() - i.min())
    return x
norm = norm1(df2)
norm

# calculating value for inertia
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# plotting scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# initialising and fitting the model for cluster 3
model = KMeans(n_clusters = 3).fit(norm)
model.labels_
cluster_labels = pd.Series(model.labels_)
df2["clust"] = cluster_labels
df2
df3 = df2.iloc[:,[47,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]]
df3
d1 = df3.iloc[:,1:].groupby(df3.clust).mean()
d1

# initialising and fitting the model for cluster 4
model = KMeans(n_clusters = 4).fit(norm)
model.labels_
cluster_labels = pd.Series(model.labels_)
df2["clust"] = cluster_labels
df2
df4 = df2.iloc[:,[47,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]]
df4
d2 = df4.iloc[:,1:].groupby(df4.clust).mean()
d2

df4.to_csv("Kmeans_autoinsurance.csv", encoding = "utf-8")

import os
os.getcwd()