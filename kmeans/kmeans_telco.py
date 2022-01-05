import pandas as pd                                                   # importing libraries
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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

# calculating value of inertia
TWSS=[]
k = list(range(2,9))
for i in k:
    kmeans =KMeans(n_clusters = i)
    kmeans.fit(norm)
    TWSS.append(kmeans.inertia_)
    
TWSS    

# plotting scree plots
plt.plot(k,TWSS,"ro-");plt.xlabel("clusters");plt.ylabel("TWSS")    
plt.show()    

# initialising and fitting the model for cluster 3
model = KMeans(n_clusters = 3).fit(norm)    
model.labels_    
    
cluster_labels = pd.Series(model.labels_)    
df2["clust"] = cluster_labels    
df3 = df2.iloc[0:,[46,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]    
df3    
d1 = df3.iloc[:,1:].groupby(df3.clust).mean()    
d1

# initialising and fitting the model for cluster 4
model = KMeans(n_clusters = 4).fit(norm)    
model.labels_    
    
cluster_labels = pd.Series(model.labels_)    
df2["clust"] = cluster_labels    
df4 = df2.iloc[0:,[46,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]    
df4    
d2 = df4.iloc[:,1:].groupby(df4.clust).mean()    
d2

df4.to_csv("Kmeans_telcochurn.csv", encoding = "utf-8")

import os
os.getcwd()