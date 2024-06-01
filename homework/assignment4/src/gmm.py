import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class GMM(object):
    
    def __init__(self, n_clusters=6):
       """
        A Gaussian mixture model trained via the expectation maximization algorithm.

        Parameters
        ----------
        n_clusters:int=6. The number of clusters / mixture components in the GMM. 

        Attributes
        ----------
        N : int. The number of examples in the training dataset.
        d : int. The dimension of each example in the training dataset.
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(C,)`
            The cluster priors.
        Q : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            The variational distribution `q(T)`.
        mu : :py:class:`ndarray <numpy.ndarray>` of shape `(C, d)`
            The cluster means.
        sigma : :py:class:`ndarray <numpy.ndarray>` of shape `(C, d, d)`
            The cluster covariance matrices.
        """
       self.parameters = {}
       self.n_clusters = n_clusters
       self.elbo = None
       self.is_fit = False
        
    def _init_params(self, X):
        """Randomly initialize the parameters of GMM model"""
        N, d = X.shape
        C = self.n_clusters
        rr = np.random.rand(C)

        self.parameters = {
            "pi": rr / rr.sum(), # cluster priors, (C, )
            "Q": np.zeros((N, C)), # variational distribution, (N, C)
            "mu": np.random.uniform(-5, 10, C * d).reshape(C, d), # cluster means, (C, d)
            "sigma": np.array([np.eye(d) for _ in range(C)]), # cluster covariance matrices
        }
        self.elbo = None
        self.is_fit = False


    def _likelihood_lowerbound(self, X):
        """Compute the lower bound of the likelihood"""
        N, d = X.shape
        P = self.parameters
        C = self.n_clusters
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]

        eps = np.finfo(float).eps # machine epsilon
        expec1, expec2 = 0.0, 0.0
        for i in range(N):
            X_i = X[i]
            for c in range(C):
                pi_k = pi[c]
                qi_k = Q[i, c]
                mu_k = mu[c, :]
                sigma_k = sigma[c, :, :]

                log_pi_k = np.log(pi_k + eps)
                log_p_xi = log_gaussian(X_i, mu_k, sigma_k)
                probs = qi_k * (log_pi_k + log_p_xi) 
                expec1 += probs
                expec2 += qi_k * np.log(qi_k + eps)
        loss = expec1 - expec2
        return expec1, expec2, loss
    
    def fit(self, X, n_iters=200, tol=1e-3, verbose=True, if_display=False):
        """
        Fit the parameters of the GMM on some training data via EM algorithm. 
       
        Returns
        -------
        success : {0, -1}
        Whether training terminated without incident (0) or one of the
        mixture components collapsed and training was halted prematurely (-1).       
        """
        prev_vlb = -np.inf 
        self._init_params(X)
        _ex1, _ex2, _vlb = [], [], []
        plt.ion()
        if if_display:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for _iter in range(n_iters):
            try:
                self._E_step(X)
                self._M_step(X)
                ex1, ex2, vlb = self._likelihood_lowerbound(X)
                
                _ex1.append(ex1)
                _ex2.append(ex2)
                _vlb.append(vlb)
                if if_display:
                    ax[0].cla()
                    ax[0].plot( _ex1, '.--', color='black')
                    ax[0].plot( _ex2, '.--', color='blue')
                    ax[0].plot( _vlb, '.--', color='red')
                    ax[0].legend(['Expectation 1', 'Expectation 2', 'Lower Bound Loss'])
                    ax[0].set_title('Lower Bound Expections')

                    ax[1].cla()
                    labels = self.predict(X, soft_labels=False)
                    ax[1].scatter(X[:, 0], X[:, 1], c=labels, s=5,\
                                   cmap=plt.cm.get_cmap('Set1', self.n_clusters))
                    ax[1].set_title(f'Iteration {_iter+1}')
                    plt.pause(0.01)
                if verbose:
                    if (_iter % 10 == 0):
                        print(f"Iter {_iter}. Lower bound:{vlb:.6f}")
                converged = _iter > 0 and np.abs(vlb - prev_vlb) <= tol
                if np.isnan(vlb) or converged:
                    break

                prev_vlb = vlb 

            except np.linalg.LinAlgError:
                print("Warning: Singular matrix, components collapsed.")
                return -1
               
        self.elbo = _vlb
        self.is_fit = True
        plt.ioff()
        if if_display:
            plt.show()
        return 0

    def predict(self, X, soft_labels=False):
        """
        Return the log probability of each data point in `X` under each mixture components.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, d)`
        soft_labels : default=False. If true, return the log probabilities of the  
        N data points. If False, only return the index of the most likely cluster. 
        """
        # assert self.is_fit, "Must call the `.fit` method before making predictions"
        
        P = self.parameters
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]
        C = self.n_clusters
        y = []

        for x_i in X:
            cprobs = [log_gaussian(x_i, mu[c, :], sigma[c, :, :]) for c in range(C)]
            if not soft_labels:
                y.append(np.argmax(cprobs))
            else:
                y.append(cprobs)
        return np.array(y)

    def _E_step(self, X):
        P = self.parameters
        C = self.n_clusters
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]
        
        for i, x_i in enumerate(X):
            denom_vals = []
            for c in range(C):
                pi_c = pi[c]
                mu_c = mu[c, :]
                sigma_c = sigma[c, :, :]

                log_pi_c = np.log(pi_c)
                log_p_xi = log_gaussian(x_i, mu_c, sigma_c)
                # log N(X_i | mu_c, sigma_c) + log pi_c
                denom_vals.append(log_pi_c + log_p_xi)
            
            # log \sum_c exp{ log N(X_i | mu_c, sigma_c) + log pi_c } 
            _max_denom = np.max(denom_vals)
            log_denom = _max_denom + np.exp(denom_vals - _max_denom).sum()
            q_i = np.exp([num - log_denom for num in denom_vals])
            # np.testing.assert_allclose(np.sum(q_i), 1, err_msg="{}".format(np.sum(q_i)))

            Q[i, :] = q_i

    def _M_step(self, X):
        N, d = X.shape
        P = self.parameters
        C = self.n_clusters
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]

        denoms = np.sum(Q, axis=0)

        # update cluster priors pi
        pi = denoms / N
        # update cluster means mu
        num_mu = [np.dot(Q[:, c], X) for c in range(C)]
        for ix, (num, den) in enumerate(zip(num_mu, denoms)):
            mu[ix, :] = num / den if den > 0 else np.zeros_like(num)
            
        # update cluster covariance matrix sigma
        for c in range(C):
            mu_c = mu[c, :]
            num_c = denoms[c]

            outer = np.zeros((d, d))
            for i in range(N):
                qc = Q[i, c]
                xi = X[i, :]
                outer += qc * np.outer(xi - mu_c, xi - mu_c)
            
            outer = outer / num_c if num_c > 0 else outer
            sigma[c, :, :] = outer
        
        # np.testing.assert_allclose(np.sum(pi), 1, err_msg="{}".format(np.sum(pi)))

""""Functions"""

def log_gaussian(x_i, mu, sigma):
    """ Computer 
    log N(x_i | mu, sigma) = log Guassian(x_i | mu, Sigma)
    = -0.5 * (log(|Sigma|) + N * log(2 * pi) + (x_i - mu)' * inv(Sigma) * (x_i - mu))
    """
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma) # 计算协方差矩阵的行列式的自然对数

    y = np.linalg.solve(sigma, x_i - mu)
    c = np.dot(x_i - mu, y) # c = (x_i - mu)' * inv(sigma) * (x_i - mu)
    return -0.5 * (a + b + c)

def load_file_data(file_path):
    X = []
    y = []
    text = np.loadtxt(file_path, skiprows=1)
    X.append(text[:, 1:])
    y.append(text[:, 0])
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

def gmm_display(X, centroids, pred_labels, likelihood, K=6):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(likelihood, '.--')
    ax[0].set_title('Likelihood Lowerbound= {:.4f}'.format(likelihood[-1]))

    ax[1].scatter(X[:, 0], X[:, 1], c=pred_labels, \
                  marker='o', s=5, cmap=plt.cm.get_cmap('Set1', K))
    ax[1].scatter(centroids[:, 0], centroids[:, 1], c=np.arange(K), \
                  marker='x', s=200, linewidths=3, cmap=plt.cm.get_cmap('Set1', K))
    ax[1].set_title('Final Iteration')
    plt.show()

if __name__ == "__main__":
    file_path = '../gmm/GMM6.txt'
    X, y = load_file_data(file_path)


    print(X.shape, y.shape)
    n_classes = int(np.max(np.unique(y))) + 1 # num of clusters or labels
    plt.scatter(X[:, 0], X[:, 1], c=y, \
                marker='o', s=5, cmap=plt.cm.get_cmap('Set1', n_classes))
    plt.colorbar()
    plt.title('Original data for clustering')
    plt.show()

    gmm = GMM(n_clusters=n_classes)
    gmm.fit(X, n_iters=200, tol=1e-3, verbose=True, if_display=True)
    labels = gmm.predict(X, soft_labels=False) # k_means predicting labels
    gmm_display(X, gmm.parameters["mu"], labels, gmm.elbo, K=n_classes)
    
    # 计算 Rand 统计量
    rand_score = metrics.adjusted_rand_score(y, labels)
    # 计算 FM 指数
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(y, labels)
    # 计算轮廓系数
    silhouette_score = metrics.silhouette_score(X, labels)

    print(f"Rand Score: {rand_score:.4f}, FM Score: {fowlkes_mallows_score:.4f}, Silhouette Score: {silhouette_score:.4f}")

