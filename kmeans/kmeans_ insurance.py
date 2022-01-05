import pandas as pd                                                 # importing libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Insurance Dataset.csv")                           # importing dataset
df
# EDA part *********************************************************************************

df.describe()                                                        # describing data set to get mean,median,sd etc
df.info()                                                            # checking datatypes and null value
plt.boxplot(df)                                                      # ploting boxplot for outliers and univariate analysis
plt.scatter(df["Premiums Paid"],df["Age"])                           # ploting scatter for bivariate analysis
plt.xlabel("Premiums Paid")
plt.ylabel("Age")
plt.scatter(df["Premiums Paid"],df["Claims made"])                   # ploting scatter for bivariate analysis
plt.xlabel("Premiums Paid")
plt.ylabel("Claims made")
plt.scatter(df["Income"],df["Claims made"])                          # ploting scatter for bivariate analysis
plt.xlabel("Income")
plt.ylabel("Claims made")
plt.matshow(df.corr())                                               # plotting plot to check correlation
plt.show()

# Data preprocessing ************************************************************************

duplicate = df.duplicated()                                          # checking duplicate values in rows                                        
duplicate.sum()

def norm1(i):                                                        # normalising the dataset to 0 and 1
    x = (i - i.min())/(i.max() - i.min())
    return x
norm = norm1(df)
norm
norm.info()

# calculating values of inertia
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters = i).fit(norm)
    TWSS.append(kmeans.inertia_)
TWSS    

# plotting scree plot
plt.plot(k,TWSS,"ro-")  ;plt.xlabel("clusters");plt.ylabel("TWSS")  

# initialising and fitting model for cluster 3
model = KMeans(n_clusters = 3).fit(norm)    
model.labels_    
    
cluster_labels = pd.Series(model.labels_)    
df["clust"] = cluster_labels        
df1 = df.iloc[0:,[5,0,1,2,3,4]]
df1
d1 = df1.iloc[0:,1:].groupby(df1.clust).mean()
d1

# initialising and fitting model for cluster 4
model = KMeans(n_clusters = 4).fit(norm)    
model.labels_    
    
cluster_labels = pd.Series(model.labels_)    
df["clust"] = cluster_labels        
df2 = df.iloc[0:,[5,0,1,2,3,4]]
df2
d2 = df2.iloc[0:,1:].groupby(df2.clust).mean()
d2

df2.to_csv("Kmeans_insurance.csv", encoding = "utf-8")

import os
os.getcwd()