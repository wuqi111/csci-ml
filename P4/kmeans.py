import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
       	'''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)
    
    	  
        center = np.take(x, np.random.choice(N,self.n_cluster), axis = 0) #choose n_cluster randomly from x
        # print(mu)
        distance_r = np.zeros(N, dtype = int)
        J = 0 
        iteration = 0
        while iteration < self.max_iter:
        	l2 = np.sum(((x - np.expand_dims(center, axis = 1))**2),axis = 2)
        	# l2 = np.linalg.norm(x - np.expand_dims(mu, axis = 1),axis = 2)
        	# niubi
        	# print(l2)
        	distance_r = np.argmin(l2,axis = 0) #np.argmin return index
        	# print(r)
        	temp =[np.sum((x[distance_r==z] - center[z])**2) for z in range(self.n_cluster)]
        	J_new = np.sum(temp)/N  # current error value
        	
        	if np.absolute(J -J_new) < self.e:
        		break
        	J = J_new
        	center = np.array([np.mean(x[distance_r == z],axis = 0) for z in range(self.n_cluster)])
        	# index = np.where(np.isnan(mu_new))
        	# mu_new[index] = mu[index]
        	# mu = mu_new
        	iteration += 1
        return (center, distance_r, iteration)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels


        def dfs(votes,membership):
        	r_ii = membership[i]
       		y_ii = y[i]
        	if y_ii not in votes[r_ii].keys(): # return all keys in dictionary
        		votes[r_ii][y_ii] = 1
        	else:
        		votes[r_ii][y_ii] += 1
        	return votes
        	

        k_means = KMeans(n_cluster = self.n_cluster , max_iter = self.max_iter , e = self.e)
        centroids, membership, interation = k_means.fit(x) # return data from the kmeans.fit
        # print(membership)
        # print(y)
        # print (self.n_cluster)
        votes = [{} for k in range(self.n_cluster)]
        # print(votes)
        # for y_i,r_i in zip(y,membership):
        for i  in range(N):
        	# each point record to the class that the point belongs to 
            votes = dfs(votes, membership)
        # print(votes)
        centroid_labels = []
        # for votes_k in votes:
        for i in range(len(votes)):
        	votes_k = votes[i]
        	if not votes_k:
        		centroid_labels.append(0)
        	value = max(votes_k,key = votes_k.get)
        	centroid_labels.append(value)
        centroid_labels = np.array(centroid_labels)
        # print(centroid_labels)

       

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py)')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        l2 = np.sum(((x - np.expand_dims(self.centroids,axis = 1))**2),axis = 2)
       	labels = self.centroid_labels[np.argmin(l2, axis = 0)]
        

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

