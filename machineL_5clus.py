#!/usr/bin/env python
# coding: utf-8

# Gépi tanulás az igék szemantikai jegyein

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('semSel.tsv',sep='\t')
df.head


# In[3]:


import random 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df_mod = pd.DataFrame()
df_mod["H"] = df["H"]
df_mod["I"] = df["I"]
df_mod["N"] = df["N"]
df_mod["A"] = df["A"]
df_mod["L"] = df["L"]
df_mod


# In[5]:


from sklearn.preprocessing import StandardScaler
X = df_mod.values
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# In[6]:


clusterNum = 5
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[7]:


df_mod["Clus_km"] = labels
df["KAT"] = labels
df_mod.groupby('Clus_km').mean()


# In[8]:


k_means_labels = k_means.labels_
k_means_labels


# In[9]:


k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# In[15]:


verbs = df['v'].tolist()

x_co = 0
y_co = 1

c = ['b+', 'r.', 'kx', 'mo', 'g*']
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(15, 10))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for i in range(len(X)):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    #my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    #cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[i][x_co], X[i][y_co], c[k_means_labels[i]])
    ax.annotate(verbs[i], (X[i][x_co], X[i][y_co]))
    
    # Plots the centroids with specified color, but with a darker outline
    #ax.plot(cluster_center[2], cluster_center[3], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('A vizsgált igék eloszlása', fontsize = 18)

# Remove x-axis ticks
#ax.set_xticks(())

# Remove y-axis ticks
#ax.set_yticks(())

plt.xlabel("HUMÁN", fontsize = 15, loc = 'right')
plt.ylabel("INTÉZMÉNY/CSOPORT", fontsize = 15, loc = 'top')

# Show the plot
plt.show()


# In[17]:


verbs = df['v'].tolist()

x_co = 0
y_co = 3

c = ['b+', 'r.', 'kx', 'mo', 'g*']
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(15, 10))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for i in range(len(X)):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    #my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    #cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[i][x_co], X[i][y_co], c[k_means_labels[i]])
    ax.annotate(verbs[i], (X[i][x_co], X[i][y_co]))
    
    # Plots the centroids with specified color, but with a darker outline
    #ax.plot(cluster_center[2], cluster_center[3], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('A vizsgált igék eloszlása', fontsize = 18)

# Remove x-axis ticks
#ax.set_xticks(())

# Remove y-axis ticks
#ax.set_yticks(())

plt.xlabel("HUMÁN", fontsize = 15, loc = 'right')
plt.ylabel("ABSZTRAKT", fontsize = 15, loc = 'top')

# Show the plot
plt.show()


# In[18]:


verbs = df['v'].tolist()

x_co = 1
y_co = 3

c = ['b+', 'r.', 'kx', 'mo', 'g*']
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(15, 10))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for i in range(len(X)):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    #my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    #cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[i][x_co], X[i][y_co], c[k_means_labels[i]])
    ax.annotate(verbs[i], (X[i][x_co], X[i][y_co]))
    
    # Plots the centroids with specified color, but with a darker outline
    #ax.plot(cluster_center[2], cluster_center[3], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('A vizsgált igék eloszlása', fontsize = 18)

# Remove x-axis ticks
#ax.set_xticks(())

# Remove y-axis ticks
#ax.set_yticks(())

plt.xlabel("INTÉZMÉNY/CSOPORT", fontsize = 15, loc = 'right')
plt.ylabel("ABSZTRAKT", fontsize = 15, loc = 'top')

# Show the plot
plt.show()


# In[14]:


df.to_csv('semSelKat5.tsv', sep="\t")


# In[20]:


k = 1
if k == 1:
    print(k)


# In[ ]:




