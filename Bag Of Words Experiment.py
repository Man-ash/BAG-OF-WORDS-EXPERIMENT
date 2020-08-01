#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.stats import multivariate_normal 
import random


# In[2]:


root = os.getcwd()
train_folder = os.path.join(root,"E:\Eck Module-3 Unsupervised Learning, Genratives Models,Pattern Discovery\IITG_PG AI&ML-03-06-2020 -Clustering using Bag of Words approach\Train_images")
train_files = os.listdir(train_folder)
data_arr = []
for i in range(len(train_files)):
    file = os.path.join(train_folder,train_files[i])
    image_array = mpimg.imread(file)
    image_patches = extract_patches_2d(image_array, (7, 7), max_patches = 100)
    for j in range(len(image_patches)):
        patch_vec = np.ravel(image_patches[j])
        data_arr.append(patch_vec)
data_arr = np.matrix(data_arr)
print(data_arr.shape)


# In[3]:


K = 3 ## K is the number of clusters that we want to create 
label_arr = np.zeros(data_arr.shape[0])
for i in range(len(label_arr)):
    label_arr[i] = np.random.choice(K)
print(label_arr)


# In[4]:


def similarity(vec1,vec2):
    
    s1 = 0
    
    vec1 = np.ravel(vec1)
    vec1 = vec1/np.linalg.norm(vec1)

    vec2 = np.ravel(vec2)
    vec2 = vec2/np.linalg.norm(vec2)
    
    L = len(vec1)
    
    for l in range(L):
        diff = vec2[l]*vec1[l]
        s1 = s1 + diff
    sim = s1
    
    return(sim)


# In[5]:


def init_mean_cov(K,data_arr,label_arr):
    mean_ls = [] ## List containing mean values of the clusters
    cov_ls = []
    size_ls = []
    cluster_ls = [[] for k in range(K)] ## Create list of empty lists to store data belonging to a certain cluster
    
    for i in range(len(label_arr)):
        for k in range(K):
            if label_arr[i] == k:  ## if the label of the data at ith row is 'k'
                norm_data = np.ravel(data_arr[i,:])/np.linalg.norm(np.ravel(data_arr[i,:]))
                cluster_ls[k].append(norm_data) ## Fill the kth empty list with this data value                
    
    for k in range(K): 
        cluster_mat = np.matrix(cluster_ls[k])
        pointNum = cluster_mat.shape[0]
        cov_k = np.cov(cluster_mat.T)
        mean_k = np.mean(cluster_mat,axis=0)
        mean_k = np.ravel(mean_k)/np.linalg.norm(np.ravel(mean_k))
        mean_ls.append(mean_k)
        cov_ls.append(cov_k)
        size_ls.append(pointNum)
    return(mean_ls,cov_ls,size_ls)    


# In[6]:


def label_update(prev_mean,data_arr,label_arr):
    for i in range(data_arr.shape[0]):
        dist_ls = [] 
        for k in range(len(prev_mean)):
            dist = similarity(data_arr[i,:],prev_mean[k]) ## Calculate the similarity of the ith datapoint with the kth mean
            dist_ls.append(dist) ## Put the distance values in a list
        dist_arr = np.array(dist_ls) ## Convert it to a NumPy array
        new_label = np.argmax(dist_arr) ##The new_label of the point is the one which is closest to the ith datapoint,i.e., it has maximum similarity
        label_arr[i] = new_label ## Set the new label
    return(label_arr)


# In[7]:


def mean_from_label(K,prev_mean,prev_cov,prev_size,data_arr,label_arr):
    cluster_ls = [[] for k in range(K)]  ## Create list of empty lists to store data belonging to a certain cluster
    
    for i in range(data_arr.shape[0]):
        for k in range(K):
            if label_arr[i] == k: ## if the label of the pixel at location [i,j] is 'k'
                norm_data = np.ravel(data_arr[i,:])/np.linalg.norm(np.ravel(data_arr[i,:]))
                cluster_ls[k].append(norm_data) ## Fill the kth empty list with this pixel value
                    
    for k in range(K):
        if len(cluster_ls[k]) !=0:  ## Only update the means of those clusters which has received at least one new point, else retain the old mean value
            cluster_mat = np.matrix(cluster_ls[k])
            pointNum = cluster_mat.shape[0]
            mean_k = np.mean(cluster_mat,axis=0)
            cov_k = np.cov(cluster_mat.T)
            mean_k = np.ravel(mean_k)/np.linalg.norm(np.ravel(mean_k))
            prev_mean[k] = mean_k
            prev_cov[k] = cov_k
            prev_size[k] = pointNum
    new_mean = prev_mean
    new_cov = prev_cov
    new_size = prev_size
    return(new_mean,new_cov,new_size)    


# In[8]:


def SphericalKMeans(data_arr,label_arr,K,maxIter):
    mean_old,cov_old,size_old = init_mean_cov(K,data_arr,label_arr)
    for t in range(maxIter):
        new_label_arr = label_update(mean_old,data_arr,label_arr)
        mean_new,cov_new,size_new = mean_from_label(K,mean_old,cov_old,size_old,data_arr,new_label_arr)
        label_arr = new_label_arr ## Update the label array
        mean_old = mean_new ## Update the mean values
        cov_old = cov_new
        size_old = size_new
        print("Iteration {} is complete during training!!".format(t+1))
    return(mean_new,cov_new,size_new)


# In[9]:


mean_new,cov_new,size_new = SphericalKMeans(data_arr,label_arr,K,20)


# In[10]:


prior_ls = size_new/np.sum(size_new)
print(prior_ls)


# In[11]:


def testImage(img_file,mean_new,cov_new,prior_ls):
    img_arr = mpimg.imread(img_file)
    img_patches = extract_patches_2d(img_arr, (7, 7), max_patches = 50)
    test_arr = []
    for i in range(len(img_patches)):
        patch_vec = np.ravel(img_patches[i])
        test_arr.append(patch_vec)
    test_arr = np.matrix(test_arr)
    print(test_arr.shape)
    for j in range(test_arr.shape[0]):
        feat_vec = []
        for k in range(len(size_new)):
            var = multivariate_normal(mean = mean_new[k],cov = cov_new[k])
            test1 = np.ravel(test_arr[j,:])
            test_sample = test1/np.linalg.norm(test1)
            lkl = var.pdf(test_sample)
            post = lkl*prior_ls[k]
            feat_vec.append(post)
        print(feat_vec/sum(feat_vec))


# In[12]:


test_folder = os.path.join(root,"E:\Eck Module-3 Unsupervised Learning, Genratives Models,Pattern Discovery\IITG_PG AI&ML-03-06-2020 -Clustering using Bag of Words approach\Test_images")
img_files = os.listdir(test_folder)
fileName = random.choice(img_files)
print(fileName)
filePath = os.path.join(test_folder,fileName)
testImage(filePath,mean_new,cov_new,prior_ls)


# In[1]:


## MANASH PRATIM KAKATI
##PG CERTIFICATION AI & ML
## E&ICT ACADAMY, IIT GUWAHATI


# In[ ]:




