import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def kmeans1(data, X, k):
  count = 0
  diff = 1
  cluster = np.zeros(X.shape[0])
  centroids = data.sample(n=k).values
  while diff:
     count += 1
     for i, row in enumerate(X):
         mn_dist = float('inf')
         for idx, centroid in enumerate(centroids):
            d = np.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
            if mn_dist > d:
               mn_dist = d
               cluster[i] = idx
     new_centroids = pd.DataFrame(X).groupby(by = cluster).mean().values
     if np.count_nonzero(centroids-  new_centroids) == 0:
        diff = 0
     else:
        centroids = new_centroids
  sns.scatterplot(X[:, 0], X[:, 1], hue = cluster)
  sns.scatterplot(centroids[:, 0], centroids[:, 1], s = 100, color = 'y')
  plt.show()
  return count, centroids, cluster



def kmeans2(data, X, k):
  count = 0
  diff = 1
  cluster = np.zeros(X.shape[0])
  centroids = data.head(n=k).values
  while diff:
     count += 1
     for i, row in enumerate(X):
         mn_dist = float('inf')
         for idx, centroid in enumerate(centroids):
            d = abs(centroid[0] - row[0]) + abs(centroid[1] - row[1])
            if mn_dist > d:
               mn_dist = d
               cluster[i] = idx
     new_centroids = pd.DataFrame(X).groupby(by = cluster).mean().values
     if np.count_nonzero(centroids - new_centroids) == 0:
        diff = 0
     else:
        centroids = new_centroids
  sns.scatterplot(X[:, 0], X[:, 1], hue = cluster)
  sns.scatterplot(centroids[:, 0], centroids[:, 1], s = 100, color = 'y')
  plt.show()
  return count, centroids, cluster



def kmeans3(data, X, k):
  count = 0
  diff = 1
  cluster = np.zeros(X.shape[0])
  centroids = data.sample(n=k).values
  while diff:
     count += 1
     for i, row in enumerate(X):
         mn_dist = float('inf')
         for idx, centroid in enumerate(centroids):
            d = max(abs(centroid[0] - row[0]), abs(centroid[1] - row[1]))
            if mn_dist > d:
               mn_dist = d
               cluster[i] = idx
     new_centroids = pd.DataFrame(X).groupby(by = cluster).mean().values
     if np.count_nonzero(centroids - new_centroids) == 0:
        diff = 0
     else:
        centroids = new_centroids
  sns.scatterplot(X[:, 0], X[:, 1], hue = cluster)
  sns.scatterplot(centroids[:, 0], centroids[:, 1], s = 100, color = 'y')
  plt.show()
  return count, centroids, cluster


def kmeans4(data, X, k):
  count = 0
  diff = 1
  cluster = np.zeros(X.shape[0])
  centroids = data.sample(n=k).values
  while diff:
     count += 1
     for i, row in enumerate(X):
         mn_dist = float('inf')
         for idx, centroid in enumerate(centroids):
            d = spatial.distance.cosine(centroid, row)
            if mn_dist > d:
               mn_dist = d
               cluster[i] = idx
     new_centroids = pd.DataFrame(X).groupby(by = cluster).mean().values
     if np.count_nonzero(centroids - new_centroids) == 0:
        diff = 0
     else:
        centroids = new_centroids
  sns.scatterplot(X[:, 0], X[:, 1], hue = cluster)
  sns.scatterplot(centroids[:, 0], centroids[:, 1], s = 100, color = 'y')
  plt.show()
  return count, centroids, cluster



def main():
    k = 6
    data = pd.DataFrame(np.random.randint(0, 100, size = (1000, 2)), columns = list('AB'))
    X = data.values
    print("Случайные центры. Евклидово расстояние:")
    count, centroids, cluster = kmeans1(data, X, k)
    print("Число итераций:", count)
    print()
    print("Центры - первые точки (по порядку). Расстояние Манхэттена:")
    count, centroids, cluster = kmeans2(data, X, k)
    print("Число итераций:", count)
    print()
    print("Случайные центры. Расстояние Чебышёва:")
    count, centroids, cluster = kmeans3(data, X, k)
    print("Число итераций:", count)
    print()
    print("Центры - первые точки (по порядку). Косинусное расстояние:")
    count, centroids, cluster = kmeans4(data, X, k)
    print("Число итераций:", count)
    data = pd.DataFrame(np.random.randint(0, 100, size = (1000, 2)), columns = list('AB'))
    X = data.values
    print("Случайные центры. Евклидово расстояние:")
    count, centroids, cluster = kmeans1(data, X, k)
    print("Число итераций:", count)
    print()
    print("Центры - первые точки (по порядку). Расстояние Манхэттена:")
    count, centroids, cluster = kmeans2(data, X, k)
    print("Число итераций:", count)
    print()
    print("Случайные центры. Расстояние Чебышёва:")
    count, centroids, cluster = kmeans3(data, X, k)
    print("Число итераций:", count)
    print()
    print("Центры - первые точки (по порядку). Косинусное расстояние:")
    count, centroids, cluster = kmeans4(data, X, k)
    print("Число итераций:", count)



    k = 6
    data = pd.DataFrame(np.array(make_circles(n_samples = 1000, shuffle = True, noise = 0.1)[0]), columns=list('AB'))
    X = data.values
    print("Случайные центры. Евклидово расстояние:")
    count, centroids, cluster = kmeans1(data, X, k)
    print("Число итераций:", count)
    print()
    print("Центры - первые точки (по порядку). Расстояние Манхэттена:")
    count, centroids, cluster = kmeans2(data, X, k)
    print("Число итераций:", count)
    print()
    print("Случайные центры. Расстояние Чебышёва:")
    count, centroids, cluster = kmeans3(data, X, k)
    print("Число итераций:", count)
    print()
    print("Центры - первые точки (по порядку). Косинусное расстояние:")
    count, centroids, cluster = kmeans4(data, X, k)
    print("Число итераций:", count)

if __name__ == "__main__":
    main()
