import pandas as pd                                       # importing libraries
from matplotlib import pyplot as plt
from feature_engine.outliers import Winsorizer 
from sklearn.cluster import KMeans
df = pd.read_csv("crime_data.csv")                        # importing dataset
df
df1 = df.drop(["Unnamed: 0"] , axis = 1)                  # dropping nominal column
df1

# EDA PART *********************************************************************************

df1.describe()                                            # describing data set to get mean,median,sd etc
df1.info()                                                # checking datatypes and null value
plt.boxplot(df1)                                          # ploting boxplot for outliers and univariate analysis
plt.scatter(df1.Murder,df1.Assault)                       # plotting scatter for bivariate analysis
plt.xlabel("Murder")
plt.ylabel("Assault")
plt.scatter(df1.Murder,df1.Rape)                          # plotting scatter for bivariate analysis
plt.xlabel("Murder")
plt.ylabel("Rape")
plt.scatter(df1.Assault,df1.UrbanPop)                     # plotting scatter for bivariate analysis
plt.xlabel("Assault")
plt.ylabel("Urbanpop")
plt.scatter(df1.UrbanPop,df1.Murder)                     # plotting scatter for bivariate analysis
plt.xlabel("Urbanpop")
plt.ylabel("Murder")
plt.matshow(df1.corr())                                  # checking for correlations
plt.show()
# Data preprocessing **************************************************************************

duplicate = df1.duplicated()                              # checking for record duplicasy
duplicate.sum()                                           # no duplicate records found
a = round(df1.Murder)                                     # converting continios to discrete by rounding
a
b = round(df1.Rape)                                       # converting continios to discrete by rounding
b
df1["Murder"] = a                                         # replacing rounded features in data set
df1["Rape"]=b
df1

# from boxplot outliers detected
# using winsorization to replace outliers in respected features with capping metod iqr and fold 1.5
# initialising winsor

winsore = Winsorizer(capping_method = "iqr",tail = "both",fold = 1.5, variables = ["Rape"])

c = winsore.fit_transform(df1[["Rape"]])                 # transforming in winsore
c
df1["Rape"] = c

df1.Rape = df1.Rape.astype("int64")                      # changing the data type float64 to int64
df1.Murder = df1.Murder.astype("int64")
df1

# using normalising technique to scale each features of dataset

def norm1(i):                                            # creating norm1 function
    x = (i-i.min())/(i.max()-i.min())
    return x
norm = norm1(df1)                                        # passing data set to function
norm

# calculating values for inertia
TWSS = []
k = list(range(2,9))
for i in  k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm)
    TWSS.append(kmeans.inertia_)
TWSS    

# plotting scree plot
plt.plot(k,TWSS,"ro-")
plt.xlabel("clusters")
plt.ylabel("inertia")

# initialising and fitting model for cluster 4
model = KMeans(n_clusters = 4).fit(norm)
model.labels_
cluster_label = pd.Series(model.labels_)
df1["clust"] = cluster_label
df1
df2 = df1.iloc[:,[4,0,1,2,3]]
df2
d1 = df2.iloc[:,0:].groupby(df2.clust).mean()
d1

# initialising and fitting model for cluster 5
model = KMeans(n_clusters = 5).fit(norm)
model.labels_
cluster_label = pd.Series(model.labels_)
df1["clust"] = cluster_label
df1
df3 = df1.iloc[:,[4,0,1,2,3]]
df3
d2 = df3.iloc[:,0:].groupby(df3.clust).mean()
d2

df3.to_csv("Kmeans_crimerate.csv", encoding = "utf-8")

import os
os.getcwd()