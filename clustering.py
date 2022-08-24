import pandas as pd #impoting pandas
import numpy as np #importing numpy
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
from scipy import spatial as spatial

original = pd.read_csv('iris.csv')
df = pd.read_csv('iris.csv') #reading pima-indians-diabetes csv file
df.pop("Species")

print("Question :- 1\n")
pca = PCA(n_components=2)#With the help of PCA method from sklearn to reduce the data
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
eg, ev = np.linalg.eig(principalDf.cov())#finding the Eigen Value and Eigen Vector and printing it
origin = [0, 0] #Setting the origin for Quiver plot
eig_vec1 = np.array([ev[0]])#Eigen Vector 1
eig_vec2 = np.array([ev[1]])#Eigen Vector 2
print("Eigen Value of Covariance Matrix : ",eg)
print("Eigen Vector of Covariance Matrix : ",ev)
print("\nScatter plot for two directions :-")#plotting the Principal components
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], alpha = 1)
plt.title("2d plot along with Eigen Vector")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.quiver(*origin, *eig_vec1[0], color="#d10af0", scale=2)
plt.quiver(*origin, *eig_vec2[0], color="#d10af0", scale=4)
plt.show()

def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

original = pd.read_csv('iris.csv')
df = pd.read_csv('iris.csv') #reading pima-indians-diabetes csv file
df.pop("Species")

pca = PCA(n_components=2).fit_transform(df)


print("Question :- 2\n")
model = KMeans(n_clusters = 3)
model.fit(pca)

labels = model.predict(pca)

plt.figure(figsize=(8, 6))
plt.scatter(pca[:,0], pca[:,1], c=model.labels_.astype(float))
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.show()
print("Distortion Measure for number of cluster = 3 is ",model.inertia_)
print("Purity score for K = 3 is ", purity(original['Species'] , labels))
print("\n")


print("Question :- 3\n")
K = [2,3,4,5,6,7]
dist_meas = {i : 0 for i in K}
purity_score = {i : 0 for i in K}
for i in K:
    model = KMeans(n_clusters = i)
    model.fit(pca)
    labels = model.predict(pca)
    dist_meas[i] = model.inertia_
    purity_score[i] = purity(original['Species'] , labels)
print("The purity score comes out to be :- ", purity_score)
plt.plot(dist_meas.keys(), dist_meas.values(), 'b-o')
plt.xlabel("Value of K")
plt.ylabel("Distortion Measure")
print("\n")

print("Question :- 4\n")
gmm = GaussianMixture(n_components=3, covariance_type='full', verbose=2,verbose_interval=3).fit(pca)
prediction_gmm = gmm.predict(pca)

centers = np.zeros((3,2))
for i in range(3):
    density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(pca)
    centers[i, :] = pca[np.argmax(density)]

print("The purity score comes out to be :- ", purity(original['Species'], prediction_gmm))
plt.figure(figsize = (10,8))
plt.scatter(pca[:, 0], pca[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6)
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.show()
print("\n")


print("Question :- 5\n")
K = [2,3,4,5,6,7]
dist_meas = {i : 0 for i in K}
purity_score = {i : 0 for i in K}
for i in K:
    gmm = GaussianMixture(n_components=i, covariance_type='full').fit(pca)
    prediction_gmm = gmm.predict(pca)
    dist_meas[i] = gmm.score(pca)
    purity_score[i] = purity(original['Species'], prediction_gmm)

print("The purity score comes out to be :- ", purity_score)
plt.plot(dist_meas.keys(), dist_meas.values(), 'b-o')
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.show()
print("\n")


print("Question :- 6\n")
dbscan_model=DBSCAN(eps=5, min_samples=4).fit(pca)

def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean): 
    # Result is noise by default 
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 # Iterate all input samples for a label 
    for j, x_new in enumerate(X_new): # Find a core sample closer than EPS 
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps: # Assign label of x_core to x_new 
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]] 
            break
    return y_new 
eps = [1, 5]
samples = [4, 10]
for i in eps:
    for j in samples:
        dbscan_model=DBSCAN(eps=i ,min_samples=j).fit(pca)
        dbtest = dbscan_predict(dbscan_model, pca, metric = spatial.distance.euclidean)
        plt.scatter(pca[:,0], pca[:,1], c=dbtest)