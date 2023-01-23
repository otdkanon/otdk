#!/usr/bin/env python
# coding: utf-8

# Gépi tanulás az igék thematikus jellemzőin

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('themSel.tsv',sep='\t')
df.head


# In[3]:


import random 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df_mod = pd.DataFrame()
df_mod["I"] = df["I"]
df_mod["N"] = df["N"]
df_mod


# In[5]:


from sklearn.preprocessing import StandardScaler
X = df_mod.values
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# In[6]:


clusterNum = 3
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


# In[10]:


# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(3, 2))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


# In[13]:


df.to_csv('themSelKat.tsv', sep="\t")


# In[ ]:




