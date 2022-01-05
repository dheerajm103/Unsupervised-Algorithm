import pandas as pd                                       # importing libraries
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

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


# calculating value for ineriia
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
df3 = df2.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
df3
d1 = df3.iloc[:,1:].groupby(df3.clust).mean()
d1

# initialising and fitting the model for cluster 4
model = KMeans(n_clusters = 4).fit(norm)
model.labels_
cluster_labels = pd.Series(model.labels_)
df2["clust"] = cluster_labels 
df4 = df2.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
df4
d2 = df3.iloc[:,1:].groupby(df4.clust).mean()
d2

df4.to_csv("Kmeans_Eastwestairlines.csv", encoding = "utf-8")

import os
os.getcwd()