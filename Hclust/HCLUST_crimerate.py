import pandas as pd                                       # importing libraries
from matplotlib import pyplot as plt
from feature_engine.outliers import Winsorizer
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

# Model building *********************************************************************************

from scipy.cluster.hierarchy import linkage              # importing libraries 
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering

# calculatingthe distance as method complete and formulae euclidean distance
z = linkage(norm, method = "complete", metric = "euclidean")
z

# plotting dendograms
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 90,  leaf_font_size = 5)
plt.show()

# using clusters as 4
# initialising and transforming AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(norm) 

h_complete.labels_                                         # creating labels in form of labels

cluster_labels = pd.Series(h_complete.labels_)             # converting cluster labels to series

df['clust'] = cluster_labels                               # taking clust to dataframe

df2 = df.iloc[:, [5,0,1,2,3,4]]                            # taking first column as clust in dataframe               
df2
d1= df2.iloc[:, 2:].groupby(df.clust).mean()               # taking means with respect to clusters
d1

# using clusters as 5
# initialising and transforming AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters= 5, linkage = "complete" , affinity = "euclidean" ).fit(norm)

h_complete.labels_                                         # creating labels in form of labels

cluster_labels = pd.Series(h_complete.labels_)             # converting cluster labels to series

df['clust'] = cluster_labels                               # taking clust to dataframe

df3 = df.iloc[:, [5,0,1,2,3,4]]                            # taking first column as clust in dataframe 
df3
d2= df3.iloc[:, 1:].groupby(df.clust).mean()               # taking means with respect to clusters
d2

df3.to_csv("HCLUST_Crimerate.csv", encoding = "utf-8")

import os
os.getcwd()

