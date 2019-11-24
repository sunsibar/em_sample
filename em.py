
import numpy as np
from sklearn.preprocessing import normalize
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import copy

from sample_data import MultVarGaussianDataGenerator as Generator




class GMixturesEM():
    def __init__(self, X, k, d=None, low=0, high=1 ):
        '''
        :param X: the data points, dimension N x d
        :param k: number of centroids
        :param d: dimension
        :param low: lower limit (on all axes) for the centroids to lie in
        :param high: higher limit (=)
        '''
        self.X = X
        self.N = len(X)
        self.k = k
        if d is None:
            d = len(X[0])
        self.d = d
        self.low = low
        self.high = high
        # initialize parameters randomly / ...
        self.centroids = np.random.random_sample(size=(self.k, self.d))*(high - low) + low
        self.cov = np.array([np.eye(self.d) * 0.01 * (high-low) for i in range(self.k)])
        self.weights = normalize((np.random.random_sample(size=(k,))+1).reshape(1,-1) , norm="l1").ravel()
        # placeholder for estimated assignment probabilities
        self.assignment_probabs = np.zeros((self.N, self.k))
        self.former_llh = 0


    def optimize(self, max_iter=2000, verbose=True):
        self.converged = False
        self.update_size = np.inf
        for counter in range(max_iter):
            if self.converged:
                break
            rvs = [stats.multivariate_normal(mean=self.centroids[c], cov=self.cov[c]) for c in range(self.k)]

            # E: find assignment-probabilities: P(ci=1 | params) ~ P(xi | ci=1,  )
            for i, point in enumerate(self.X):
                # P(xi <-> cj | params) =  P(xi | cj, params) * P(cj | params) / (Z)  -- right?
                old_assignment_prob = copy.copy(self.assignment_probabs)
                self.assignment_probabs[i,:] = [rvs[j].pdf(point)*self.weights[j] for j in range(self.k)]
                #self.assignment_probabs[i,:] = [rvs[j].pdf(point) for j in range(self.k)]
                self.assignment_probabs[i, :] = normalize(self.assignment_probabs[i, :].reshape(1,-1), norm="l1").ravel()
                self.update_size = np.linalg.norm(self.assignment_probabs - old_assignment_prob)
                self.converged = (np.linalg.norm(self.assignment_probabs - old_assignment_prob) <=
                                  (self.high - self.low)*len(self.X)*self.d * 0.00000001)

            # Test: what's the total data likelihood:

            llh = np.product([
                np.sum([rvs[j].pdf(point) * self.weights[j] for j in range(self.k)])
                for i, point in enumerate(self.X)
            ])
            if verbose:
                print("data likelihood:\n  ", llh)
                if llh<self.former_llh:
                    print("* "*10)
                    print("Error: LLH decreased from ", self.former_llh, " to ", llh)
            assert llh >= self.former_llh, "Error: LLH decreased from " +str(self.former_llh) + " to " + str(llh)
            self.former_llh = llh
            # compared to when using assignment probabilities:
            llh = np.product([
                np.sum([rvs[j].pdf(point) * self.assignment_probabs[i,j] for j in range(self.k)])
                for i, point in enumerate(self.X)
            ])
            if verbose:
                print("data likelihood using assignment probabs:\n ", llh)

            # M: find new parameter estimatfes:
            for j in range(self.k):
                assert  np.sum(self.assignment_probabs[:, j]) >= 0.00001, "how can the algorithm not use a centroid at all?"
            self.weights = np.array([1/self.N * np.sum(self.assignment_probabs[:, j])   for j in range(self.k)  ])
            assert np.isclose(np.sum(self.weights) , 1)
            assert np.all(np.isclose(normalize(self.weights.reshape(1,-1), norm="l1"), self.weights))
            # cov[j] = (sum_i ass_ij * (xi - mj)(xi - mj)^T ) / (sum assignments j)
            self.cov = np.array([
                np.sum([ self.assignment_probabs[i, j] *
                          np.outer(self.X[i,:] - self.centroids[j,:], self.X[i,:] - self.centroids[j,:])
                          for i in range(self.N)], axis=0)   / (np.sum(self.assignment_probabs[:, j]) ) + np.eye(self.d) * + 0.000001
              for j in range(self.k)])
            self.centroids = np.array([
                np.sum([val * vec for val, vec in zip(self.assignment_probabs[:,j], self.X )], axis=0) / ( np.sum(self.assignment_probabs[:, j]))
              for j in range(self.k)])
            if verbose:
                print("cov norms:")
                for j in range(self.k):
                    print(np.linalg.norm(self.cov[j]))

        return self.converged


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib'''
    return plt.cm.get_cmap(name, n)

def main(display=True):
    k = 9
    d = 3
    centroids = np.random.random_sample(size=(k, d))
    weights = normalize(np.random.random_sample(size=(k,)).reshape(1,-1) , norm="l1").ravel()
    cov = [np.eye(d)*0.01 for _ in range(k)]
    gen = Generator(centroids, cov, weights, k=k, d=d)

    samples, C = gen.sample(100)

    # plot the samples and true centroids:
    cmap = get_cmap(k+2)
    if display:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(*[centroids[:,i] for i in range(d)], marker='o', color='k' ) # centroids[:,0], centroids[:,1], centroids[:,2])
        for j in range(k):
            j_points = np.where(C == j)
            ax.scatter(*[samples[j_points, i] for i in range(d) ], marker='^', color=cmap(j), alpha=0.5 ) # centroids[:,0], centroids[:,1], centroids[:,2])
        ax.set_xlim([-0.5,1.5])
        ax.set_ylim([-0.5,1.5])
        ax.set_zlim([-0.5,1.5])
        plt.draw()
        plt.pause(0.001)

    # do EM
    em = GMixturesEM(samples, k, d, low=0, high=1)
    converged = False
    for counter in range(1000):
        # Show the result every 10 steps
        if display:
            if "fig1" in globals():
                fig1.clf()
            else:
                fig1 = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(*[em.centroids[:, i] for i in range(em.d)], marker='o',
                       color='r')  # centroids[:,0], centroids[:,1], centroids[:,2])

            ax.scatter(*[em.X[:, i] for i in range(em.d)], marker='^', color='b',
                       alpha=0.5)  # centroids[:,0], centroids[:,1], centroids[:,2])
            ax.scatter(*[centroids[:, i] for i in range(d)], marker='o',
                       color='k')  # centroids[:,0], centroids[:,1], centroids[:,2])
            ax.set_xlim([-0.5, 1.5])
            ax.set_ylim([-0.5, 1.5])
            ax.set_zlim([-0.5, 1.5])
            plt.draw()
            plt.pause(0.001)
            # compare assignments to the starting plot

            if "fig2" in globals():
                fig2.clf()
            else:
                fig2 = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(*[em.centroids[:, i] for i in range(em.d)], marker='x',
                       color='r')  # centroids[:,0], centroids[:,1], centroids[:,2])
            for j in range(k):
                j_points = np.where(np.argmax(em.assignment_probabs, axis=1) == j)
                ax.scatter(*[samples[j_points, i] for i in range(d)], marker='^', color=cmap(j),
                           alpha=0.5)  # centroids[:,0], centroids[:,1], centroids[:,2])
            ax.set_xlim([-0.5, 1.5])
            ax.set_ylim([-0.5, 1.5])
            ax.set_zlim([-0.5, 1.5])
            plt.draw()
            plt.pause(0.001)


        if converged:
            print("converged")
            print("centroids vs true centroids:")
            for j in range(k):
                print(centroids[j])
            print("^-- true centroids")
            print("guessed centroids --v")
            for j in range(k):
                print(em.centroids[j])
            break

        converged = em.optimize(1, verbose=True)







if __name__ == '__main__':
    np.random.seed(33)  # 22: works; 2: strange, but not wrong, 24: good if convergence criterion strict; 124: wrong but not wrong
    # Failure cases: seed = 33, both for k=2 and k=3.
    import traceback
    num_errors = 0
    for i in range(20):
        try:
            main(display=False)
        except AssertionError as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            print("Assertion error")
            num_errors += 1
    print("number of assertion errors: ", num_errors)