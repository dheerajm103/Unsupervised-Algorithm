import pandas as pd                                                      # importing libraries
from matplotlib import pyplot as plt

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

# Model building
from scipy.cluster.hierarchy import linkage                              # importing libraries 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

z = linkage(norm, method = "complete" , metric = "euclidean")            # calculating distance
z

# plotting dendograms
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z , leaf_rotation = 0 , leaf_font_size = 10)
plt.show()

# initialising and fitting in model for cluster 3
h_complete = AgglomerativeClustering(n_clusters = 3 , linkage = "complete", affinity = "euclidean").fit(norm)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df2["clust"] = cluster_labels 
df3 = df2.iloc[:, [47,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]]                            # taking first column as clust in dataframe               
df3
d1= df3.iloc[:, 1:].groupby(df3.clust).mean()               # taking means with respect to clusters
d1

# initialising and fitting in model for cluster 4
h_complete = AgglomerativeClustering(n_clusters = 4 , linkage = "complete", affinity = "euclidean").fit(norm)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df2["clust"] = cluster_labels 
df4 = df2.iloc[:, [47,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]]                            # taking first column as clust in dataframe               
df4
d2= df4.iloc[:, 1:].groupby(df4.clust).mean()               # taking means with respect to clusters
d2

df4.to_csv("HCLUST_Autoinsurance.csv", encoding = "utf-8")

import os
os.getcwd()