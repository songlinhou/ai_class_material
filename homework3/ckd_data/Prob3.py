#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:37:27 2019

@author: Ray
"""
from __future__ import print_function
from sklearn.datasets import load_digits
from Prob1 import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def relabel_based_on_majority_labels(predicted_labels,ground_truth, transform=True):
  """
  This function relabels the cluster with the most frequent digit in that cluster
  """
  unique_predicted_labels = np.unique(predicted_labels)
  bool_array_list = []
  most_frequent_label_list = []
  for cluster_id in unique_predicted_labels:
    bool_array_list.append(predicted_labels == cluster_id)
  
  for bool_array in bool_array_list:
    ground_truth_labels = ground_truth[bool_array]
    labels_in_group, label_count = np.unique(ground_truth_labels,return_counts=True)
    most_frequent_label = labels_in_group[np.argmax(label_count)]
    if most_frequent_label not in most_frequent_label_list:
      most_frequent_label_list.append(most_frequent_label)
    else:
      vc = pd.value_counts(ground_truth_labels)
      vc_index = np.array(vc.index)
      # print(vc_index)
      find_frequent_label = False
      for label in vc_index:
        if (label in most_frequent_label_list):
          continue
        else:
          find_frequent_label = True
          most_frequent_label = label
          break
      if find_frequent_label:
        most_frequent_label_list.append(most_frequent_label)
      else:
        # just pick one from the remaining
        remaining =  set(ground_truth) - set(most_frequent_label_list)
        if(len(remaining) == 0):
          continue
        most_frequent_label_list.append(list(remaining)[0])
        
        
  transformed_output  = np.ones(predicted_labels.shape) * -1
  if transform:
    for bool_array,freq_label in zip(bool_array_list,most_frequent_label_list):
      transformed_output[bool_array] = freq_label
    return transformed_output
  return most_frequent_label_list


def relabel_and_merge(y_pred,target):
  all_clusters_ids = np.unique(y_pred)
  digit_to_cluster_map = {}
  for cluster_id in all_clusters_ids:
    # print("for cluster id: {}".format(cluster_id))
    digits_in_cluster = target[y_pred == cluster_id]
    most_freq_num_in_cluster = pd.Series(digits_in_cluster).value_counts().index[0]
    if most_freq_num_in_cluster not in digit_to_cluster_map.keys():
      digit_to_cluster_map[most_freq_num_in_cluster] = []
    digit_to_cluster_map[most_freq_num_in_cluster].append(y_pred == cluster_id)
  
  y_relabelled = np.ones(y_pred.shape) * -1
  for digit in digit_to_cluster_map.keys():
    clusters_with_digit = digit_to_cluster_map[digit]
    for record_index in clusters_with_digit:
      y_relabelled[record_index] = digit
  
  return y_relabelled




digits = load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
n_samples = len(digits.images)
img_data = digits.images.reshape((n_samples, -1))


# try the k-means implemented in Prob1.py
k = 10
k_means_digits = KMeans(img_data,10)
k_means_results = k_means_digits.fit(verbose=False)

# use AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=10)
ac.fit(img_data)
ac_results = ac.labels_

# use AffinityPropagation
ap = AffinityPropagation()
ap_results = ap.fit_predict(img_data)

# relabeling 
k_means_output = relabel_based_on_majority_labels(k_means_results,digits.target)
ac_output = relabel_based_on_majority_labels(ac_results,digits.target)
ap_output = relabel_and_merge(ap_results,digits.target)


# display the confusion matrice

confusion_matrix_k_means = confusion_matrix(digits.target,k_means_output)
print("confusion matrix of k_means is:")
print(confusion_matrix_k_means)

confusion_matrix_ac = confusion_matrix(digits.target,ac_output)
print("confusion matrix of Agglomerative Clustering is:")
print(confusion_matrix_ac)

confusion_matrix_ap = confusion_matrix(digits.target,ap_output)
print("confusion matrix of Affinity Propagation Clustering is:")
print(confusion_matrix_ap)

score_k_means = metrics.fowlkes_mallows_score(digits.target, k_means_output) 
score_ac = metrics.fowlkes_mallows_score(digits.target, ac_output) 
score_ap = metrics.fowlkes_mallows_score(digits.target, ap_output) 

print("The Fowlkes Mallows Scores of these 3 methods are:")
print("K-means: {}".format(score_k_means))
print("Agglomerative Clustering: {}".format(score_ac))
print("Affinity Propagation: {}".format(score_ap))
