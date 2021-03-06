import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, membership, iteration = k_means.fit(x)
            membership = np.identity(self.n_cluster)[membership]
            
            n = np.sum(membership, axis=0)
            self.variances = np.zeros((self.n_cluster, D, D))
            for i in range(self.n_cluster):
                x_k = x - self.means[i]
                
                self.variances[i] = np.dot(x_k.T * membership[:, i], x_k) / n[i]

            self.pi_k = n / N

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            n = np.zeros(self.n_cluster)
            self.means = np.random.uniform(0, 1, (self.n_cluster, D))
            self.variances = np.array([np.eye(D)] * self.n_cluster)
            self.pi_k = np.full(self.n_cluster, 1 / self.n_cluster)
            
            # Initialize gamma
            def sign(sigma):
                rank = np.linalg.matrix_rank(sigma)
                while rank < D:
                    sigma += np.eye(D) * 0.001
                    rank = np.linalg.matrix_rank(sigma)
                det = np.linalg.det(sigma)
                denom = np.sqrt((2 * np.pi) ** D * det)
                return (denom,sigma)
            membership = np.zeros((N, self.n_cluster))
            
            for k in range(self.n_cluster):
                mu = self.means[k]
                sigma = np.copy(self.variances[k])
                tup = sign(sigma)
                sigma = tup[1]
                denom = tup[0]
                f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1)) / denom
                membership[:, k] = self.pi_k[k] * f

            membership = (membership.T / np.sum(membership, axis=1)).T



            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        def sign(sigma):
            rank = np.linalg.matrix_rank(sigma)
            while rank < D:
                sigma += np.eye(D) * 0.001
                rank = np.linalg.matrix_rank(sigma)
            det = np.linalg.det(sigma)
            denom = np.sqrt((2 * np.pi) ** D * det)
            return (denom,sigma)

        l = -np.inf
        iteration = 0
        while iteration < self.max_iter:
            iteration += 1
            # E step
            for k in range(self.n_cluster):
                mu = self.means[k]
                sigma = np.copy(self.variances[k])
                tup = sign(sigma)
                sigma = tup[1]
                denom = tup[0]
                f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1)) / denom
                membership[:, k] = self.pi_k[k] * f
            
            
            l_new = np.sum(np.log(np.sum(membership, axis=1)))
            
            
            membership = (membership.T / np.sum(membership, axis=1)).T
            n = np.sum(membership, axis=0)
            
            
            for k in range(self.n_cluster):
                self.means[k] = np.sum(membership[:, k] * x.T, axis=1).T / n[k]
                self.variances[k] = np.dot((x - self.means[k]).T * membership[:, k], x - self.means[k]) / n[k]
            self.pi_k = n / N
            
            if np.abs(l - l_new) <= self.e:
                break
            l = l_new
        return iteration


        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        D = self.means.shape[1]
        
        zzz = np.random.choice(self.n_cluster, N, p=self.pi_k)
        samples = np.zeros((N, D))
        # for i, k in enumerate(z):
        for i in range(len(zzz)):
            k = zzz[i]
            mu = self.means[k]
            sigma = self.variances[k]
            samples[i] = np.random.multivariate_normal(mu, sigma)
        


        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)


        def sign(sigma):
            rank = np.linalg.matrix_rank(sigma)
            while rank < D:
                sigma += np.eye(D) * 0.001
                rank = np.linalg.matrix_rank(sigma)
            det = np.linalg.det(sigma)
            denom = np.sqrt((2 * np.pi) ** D * det)
            return (denom,sigma)

        N, D = x.shape
        l = 0
        joint = np.zeros((N, self.n_cluster))

        for k in range(self.n_cluster):
            mu = self.means[k]
            sigma = np.copy(self.variances[k])
            tup = sign(sigma)
            sigma = tup[1]
            denom = tup[0]
            f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1)) / denom
            joint[:, k] = self.pi_k[k] * f

        log_likelihood = float(np.sum(np.log(np.sum(joint, axis=1))))
        


        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            raise Exception('Impliment Guassian_pdf __init__')
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            raise Exception('Impliment Guassian_pdf getLikelihood')
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
