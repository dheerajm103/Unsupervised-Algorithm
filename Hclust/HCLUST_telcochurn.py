import pandas as pd                                                   # importing libraries
from matplotlib import pyplot as plt


df = pd.read_csv("Telco_customer_churn.csv")                          # importing dataset
df
df1 = df.drop(["Customer ID","Count","Quarter","Referred a Friend","Internet Service","Paperless Billing"], axis = 1)
df1                                                                   # dropping nominal column
df2 = pd.get_dummies(df1 )                                            # getting dummies column to char features
df2
# EDA part *********************************************************************************

df2.describe()                                                        # describing data set to get mean,median,sd etc
df1.info()                                                            # checking datatypes and null value
plt.boxplot(df2)                                                     # ploting boxplot for outliers and univariate analysis
plt.scatter(df2["Total Revenue"],df2["Total Long Distance Charges"]) # ploting scatter for bivariate analysis
plt.xlabel("Total Revenue")
plt.ylabel("Total Long Distance Charges")
plt.scatter(df2["Total Revenue"],df2["Monthly Charge"])
plt.xlabel("Total Revenue")
plt.ylabel("Monthly Charges")
plt.scatter(df2["Total Revenue"],df2["Tenure in Months"])
plt.xlabel("Total Revenue")
plt.ylabel("Tenure in Months")
plt.matshow(df1.corr())                                               # plotting plot to check correlation
plt.show()

# Data preprocessing ************************************************************************

duplicate = df1.duplicated()                                          # checking duplicate values in rows                                        
duplicate.sum()

def norm1(i):                                                        # normalising the dataset to 0 and 1
    x = (i - i.min())/(i.max() - i.min())
    return x
norm = norm1(df2)
norm
norm.info()

# Model building
# importing libraries

from scipy.cluster.hierarchy  import linkage                        
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# calculating  distances
z = linkage(norm,method = "complete",metric = "euclidean")
z

# plotting dendograms
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,  leaf_font_size = 10)
plt.show()

# initialising and fitting the model for cluster 3
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(norm) 

h_complete.labels_                                         # creating labels in form of labels

cluster_labels = pd.Series(h_complete.labels_)             # converting cluster labels to series

df2['clust'] = cluster_labels                               # taking clust to dataframe

df3 = df2.iloc[:, [46,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]                            # taking first column as clust in dataframe               
df3
d1= df3.iloc[:, 2:].groupby(df3.clust).mean()               # taking means with respect to clusters
d1

# initialising and fitting the model for cluster 6
h_complete = AgglomerativeClustering(n_clusters = 6, linkage = 'complete', affinity = "euclidean").fit(norm) 

h_complete.labels_                                         # creating labels in form of labels

cluster_labels = pd.Series(h_complete.labels_)             # converting cluster labels to series

df2['clust'] = cluster_labels                               # taking clust to dataframe

df4 = df2.iloc[:, [46,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]                            # taking first column as clust in dataframe               
df4
d2= df4.iloc[:, 2:].groupby(df4.clust).mean()               # taking means with respect to clusters
d2

df4.to_csv("HCLUST_Telcochurn.csv", encoding = "utf-8")

import os
os.getcwd()