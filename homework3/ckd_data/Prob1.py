import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class KMeans():
  def __init__(self,dataset,k):
    self.dataset = pd.DataFrame(dataset)
    self.k = k
  
  def dist(self,x0,y0,x1,y1):
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)

  def get_dist_of_vectors(self,vector1,vector2,use_sqrt=True):
    diff = np.array(vector1) - np.array(vector2)
    diff_squared = diff ** 2
    res = np.sum(diff_squared)
    if use_sqrt:
      return np.sqrt(res)
    return res
  
  def get_dist_of_other_points(self,row):
    center_points_values = self.center_point_values
#    center_num = len(center_points_values)
    dists = []
    row_value = row.values
    # print('center_points_values=',center_points_values)
    for center_id,center_point in enumerate(center_points_values):
      distance = self.get_dist_of_vectors(center_point,row_value)
      dists.append(distance)
    chosen_cluster_id = np.argmin(dists)
    return chosen_cluster_id
  
  def compute_new_centers(self,clustered_ids):
    avgs = []
    df = self.dataset
    k = self.k
    for i in range(k):
      points_of_cluster_i = df[clustered_ids == i]
      avg_point_for_ci = np.average(points_of_cluster_i,axis=0)
      avgs.append(avg_point_for_ci)
    return np.array(avgs)
    
  def fit(self,random_seed = None,max_iters = 100,verbose=True):
    df = self.dataset
    if random_seed is not None:
      np.random.seed(random_seed)
    self.init_central_indexes = np.random.choice(range(len(df)),self.k,replace=False)
    self.center_points = df.iloc[self.init_central_indexes]
    self.center_point_values = self.center_points.values
    # print(self.center_point_values)
    if verbose:
      print('-'*20)
      print("the init centers are:")
      print(self.center_point_values)
    self.clustered_result = self.dataset.apply(self.get_dist_of_other_points,axis=1)
#     print(self.clustered_result)
#     clustered_result_copy = np.copy(self.clustered_result) # get the clustering result
    
    
    next_clustered_result = np.zeros(self.clustered_result.shape)
    iters = 0
    while True:
      if iters > max_iters:
        if verbose:
          print("maximum iterations reached!")
        break
      self.center_point_values = self.compute_new_centers(self.clustered_result) # update centers  
      chosen_cluster_id = self.dataset.apply(self.get_dist_of_other_points,axis=1)
      
      if np.all(chosen_cluster_id == self.clustered_result):
        if verbose:
          print('-'*20)
          print("finish! at iter {}".format(iters))
        break
      else:
        self.clustered_result = chosen_cluster_id
      iters += 1
#     print("program finished")
    return chosen_cluster_id

  def get_centers(self):
    return self.center_point_values

def plot_original_data(df):
    plt.plot(df['x'],df['y'],'.')
    plt.xlabel('Length')
    plt.ylabel("Width")
    plt.title("The initial dataset")

def plot_results(df,k):
    kmeans = KMeans(df,k)
    clusterd_result = kmeans.fit(verbose=True)
    centers = kmeans.get_centers()
    for id in range(k):
        cluster_i = df[clusterd_result == id]
        #   print(cluster_i)
        plt.plot(cluster_i['x'],cluster_i['y'],'.')

    for point in centers:
        plt.plot(point[0],point[1],'o',markersize=10)
    
    plt.xlabel('Length')
    plt.ylabel("Width")
    plt.title("The dataset clustered with k==2")

if __name__ == '__main__':
    df = pd.read_csv('cluster_data.txt',names=['x','y'],sep='\t')
    plot_original_data(df)
    plot_results(df,2)
