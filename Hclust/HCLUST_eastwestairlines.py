import pandas as pd                                       # importing libraries
import matplotlib.pylab as plt

df = pd.read_csv("EastWestAirlines.csv")                  # importing dataset
df["ID"]
df1 = df.drop(["ID"], axis = 1)                           # dropping ID column
df1

#  EDA Part ******************************************************************************************
df1.describe()                                            # describing data set to get mean,median,sd etc
df1.info()                                                # checking datatypes and null value
plt.boxplot(df1)                                          # ploting boxplot for outliers and univariate analysis
plt.scatter(df1.Balance,df1.Bonus_miles)                  # plotting scatter for bivariate analysis
plt.xlabel("Balance")
plt.ylabel("Bonus_miles")
plt.scatter(df1.Balance,df1.Days_since_enroll)            # plotting scatter for bivariate analysis
plt.xlabel("Balance")
plt.ylabel("Days_since_enroll")
plt.scatter(df1.Bonus_miles,df1.Days_since_enroll)        # plotting scatter for bivariate analysis
plt.xlabel("Bonus_miles")
plt.ylabel("Days_since_enroll")
plt.matshow(df1.corr())                                   # checking for correlations
plt.show()
# Data preprocessing and Feature engineering *********************************************************

duplicate = df1.duplicated()                              # checking for record duplicasy
duplicate.sum()                                           # one duplicate record found
df2 = df1.drop_duplicates()                               # removed duplicate record
df2



# using normalising technique to scale each features of dataset
def  norm1(i):                                           # defining norm1 function to scale data to 0 & 1
    x = (i -i.min())/(i.max()-i.min())
    return x
norm = norm1(df2)                                       # passing dataset set after cleansing                                   
norm                                                    # obtained scaled dataset but three columns as nan value

# Model building ********************************************************************************************

from scipy.cluster.hierarchy import linkage             # importing libraries
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering

# calculating distance metod as complete and formulae as euclidean
z = linkage(norm, method = "complete", metric = "euclidean")   
z

# plotting dendograms
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,  leaf_font_size = 10)
plt.show()

# using clusters as 3
# initialising and transforming AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(norm) 

h_complete.labels_                                     # created labels in form of array                                  
cluster_labels = pd.Series(h_complete.labels_)         # converting labels to series

df['clust'] = cluster_labels                           # taking clust column to dataframe

df3 = df.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]       # in dataframe taking 1st column as clust
df3
d1 = df3.iloc[:, 2:].groupby(df.clust).mean()          # calculating mean with respect to clust column
d1

# using clusters as 4
# initialising and transforming AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(norm) 

h_complete.labels_                                     # created labels in form of array 

cluster_labels = pd.Series(h_complete.labels_)         # converting labels to series

df['clust'] = cluster_labels                           # taking clust column to dataframe

df4 = df.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]       # in dataframe taking 1st column as clust
df4
d2= df4.iloc[:, 2:].groupby(df.clust).mean()           # calculating mean with respect to clust column
d2

df4.to_csv("HCLUST_Eastwestairlines.csv", encoding = "utf-8")

import os
os.getcwd()