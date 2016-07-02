# coding: utf-8

"""
Created on Tue July 01 14:56:24 2016

@author: Chris
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

#from IPython.display import display
#get_ipython().magic(u'matplotlib inline')

#bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pd.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pd.read_csv('D:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pd.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pd.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with 'No Morphology'
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ','No Morphology')

#Remove any data with NaN values
data2 = data.dropna()
data2.describe()

#For K-Means cluster analysis, we don't usually take a lot of categorical variables

def cratermorph(x):
    if x == 'No Morphology':
        return 0
    else:
        return 1
    
data2['CRATER_MORPHOLOGY_BIN'] = data2['MORPHOLOGY_EJECTA_1'].apply(cratermorph)
data2['CRATER_MORPHOLOGY_BIN'] = data2['CRATER_MORPHOLOGY_BIN'].astype('category')
data2.head(5)

#We now set our clustering variables
cluster = data2[['LATITUDE_CIRCLE_IMAGE','LONGITUDE_CIRCLE_IMAGE','DEPTH_RIMFLOOR_TOPOG',
                 'NUMBER_LAYERS','CRATER_MORPHOLOGY_BIN']]

#Standardize clustering variables to have mean=0 and sd=1
clustervar = cluster.copy()
for a in cluster:
    clustervar[a] = preprocessing.scale(clustervar[a].astype('float64'))

#split the data into training and test sets
clus_train, clust_test = train_test_split(clustervar, test_size=.3, random_state=123)

#we'll run a cluster analysis for 1-10 clusters as recommended on the and create an empty array
#where we'll place the average of the minimum euclidean distant of each point from our cluster centers
clusters = range(1,11)
meandist = []

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis = 1)) / clus_train.shape[0])

#Plot the average distance from observations from the cluster centroids and use elbow method to identify number of clusters
#we should use in our analysis

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting K with the Elbow Method')

#The above plot shows that there may be about 5 clusters...so let's look at these

model5 = KMeans(n_clusters=5)
model5.fit(clus_train)
clusassign=model5.predict(clus_train)

#Plot the clusters
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model5.labels_)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 5 Clusters')
plt.show()

#We'll now merge cluster assignment with clustering variables to look at cluster variable means by cluster

#creating a unique identifier variable from the index for cluster training data to merge with cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
#create a list that has the new index variable
cluslist=list(clus_train['index'])
#create a list of cluster assignments
labels=list(model5.labels_)
#combine index variable list with cluster assignment list into a dictionary
newlist = dict(zip(cluslist,labels))
#convert newlist dictionary to a dataframe
newclus2 = DataFrame.from_dict(newlist,orient='index')
newclus2.columns = ['cluster']

#now for the cluster assignment variable, we'll reset the index for to create a key column so
#we can merge the dataframe with the training data

newclus2.reset_index(level=0, inplace=True)

#merging the cluster assignment dataframe
merged_train=pd.merge(clus_train, newclus2, on='index')
#cluster frequencies
merged_train.cluster.value_counts()

#Now we'll calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print('Clustering variable means by cluster')
print(clustergrp)

#We'll merge back the latitude data and examine whether there are indeed difference in clusters after running an ANOVA

responsevar = data2['DIAM_CIRCLE_IMAGE']

#we now split this into training and test sets

response_train, response_test = train_test_split(responsevar, test_size=.3, random_state=123)
response_train1 = DataFrame(response_train)
response_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(response_train1, merged_train, on='index')
merged_train_all_sub = merged_train_all[['DIAM_CIRCLE_IMAGE','cluster']].dropna()

latitudemodel = smf.ols(formula='DIAM_CIRCLE_IMAGE ~ C(cluster)',data=merged_train_all_sub).fit()
print(latitudemodel.summary())

print('means and standard deviations for Crater Diameter by cluster')
m1 = merged_train_all_sub.groupby('cluster').mean()
m2 = merged_train_all_sub.groupby('cluster').std()
m1.columns=['DIAM_CIRCLE_IMAGE_MEAN']
m2.columns=['DIAM_CIRCLE_IMAGE_STDEV']
m3 = pd.merge(m1,m2,left_index=True,right_index=True)
print(m3)

mc1 = multi.MultiComparison(merged_train_all_sub['DIAM_CIRCLE_IMAGE'],merged_train_all_sub['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())

#Had too much trouble trying to get legend labels on the pyplot so used seaborn instead
plot_columns = DataFrame(pca_2.fit_transform(clus_train))
model5labels = DataFrame(model5.labels_)
model5labels.columns = ['cluster']
plot_columns.columns = ['CV1','CV2']
model5labels.reset_index(level=0,inplace=True)
plot_columns.reset_index(level=0,inplace=True)

plot_columns2 = pd.merge(plot_columns,model5labels,on='index')
plot_columns2.head(5)

#plot data using seaborn to see which clusters are labeled what
import seaborn
seaborn.lmplot(x='CV1',y='CV2',data=plot_columns2,hue='cluster',fit_reg=False)

