from sklearn.cluster import KMeans
import numpy as np

def avg_distance(pt, pts) :
	"""function to calculate the average distance between a point and cluster

	Parameters
	----------
	pt : float
		central point for distance calculation
	pts : array-like
		cluster of points for distance calculation

	Returns
	-------
	float
		average distance from point to clusters
	"""
	if(pts.shape[0] == 0) :																			# in case of empty array
		return np.nan																				# tends to 0
	return np.mean(abs(pts - pt))																	# average distance calculation

def Silhouette_value(features , labels , k) :
	"""function to calculate the silhouette value for the cluster

	Parameters
	----------
	features : 2D array
		intersities i.e. point for k-mean clustersing
	labels : 1D array
		labels of the datapoints (i.e. features) provided
	k : int
		number of clusters or labels

	Returns
	-------
	float
		silhoute score fo the cluster
	"""
	# k = labels.max() + 1
	silhouette_coeffs = []
	for i in range(features.shape[0]) :
		neighbour_labels = labels.copy()
		neighbour_labels[i] = k																		# labels - {current datapoint}
		A = avg_distance(features[i] , features[neighbour_labels == labels[i]])						# avg distance in same cluster
		B = np.inf																					# avg distance with neigbouring cluster
		for j in range(k) :
			if(j != labels[i]) :
				temp_B = avg_distance(features[i] , features[neighbour_labels  == j])
				B = min(temp_B, B)																	# minimizing avg distance with neigbouring cluster
		S = (B - A) / max(A,B)
		silhouette_coeffs.append(np.nan_to_num(S))													# appending silhoutte coefficient
	# print(k , silhouette_coeffs)
	return np.mean(silhouette_coeffs)																# returning silhoutte score

def optimizing_k(img_features) :
	"""functionn to optimize the k for k-mean cluster

	Parameters
	----------
	img_features : 2D array
		feature array of datapoints

	Returns
	-------
	int
		optimized k
	"""
	vals = []
	for k in range(2, img_features.shape[0]-1) :
		kmeans = KMeans(n_clusters=k).fit(img_features)												# fitting k-mean cluster model
		val = Silhouette_value(img_features , kmeans.labels_ , k)									# calculating silhouette score
		print('Silhouette score for max' , k , 'clusters is' , val)
		vals.append(val)
	
	return np.argmax(vals)+2

if __name__ == '__main__' :
	img = np.array([[10,5,1],[4,10,2],[11,1,12]])
	X = img.reshape(-1).reshape(-1,1)

	optimized_K = optimizing_k(X)
	print('number of Optimal cluster is' , optimized_K)



