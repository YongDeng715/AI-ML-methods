# Implementing NaviveBayes Classifier
import numpy as np

'''
(word_matrix) X_counts: (num_samples, num_labels), y: (num_samples,)
bernulli:    word_matrix[i, label_mapping[word]] = 1
multinomial: word_matrix[i, label_mapping[word]] = count

multinomial naive bayes:

bernulli naive bayes:

gaussian naive bayes:

'''

class MultinomialNBC(object):
    def __init__(self):
        self.class_prior_ = None # (n_classes,)
        self.ftr_log_prob_ = None 
        self.classes = None # (n_classes,)
        self.ftr_counts = None # (n_classes, n_features)
        self.n_classes = None 

    def fit(self, X_counts, y):
        self.classes = np.unique(y) # get unique classes
        self.n_classes = len(self.classes) # num of classes

        self.class_counts = np.zeros(self.n_classes)
        self.ftr_counts = np.zeros((self.n_classes, X_counts.shape[1]))

        for i, cls_ in enumerate(self.classes):
            cls_indices = np.where(y == cls_)[0] # get indices of class cls_
            self.class_counts[i] = len(cls_indices)
            self.ftr_counts[i] = np.array(X_counts[cls_indices].sum(axis=0)).flatten()

        # calculate class priors and feature log probabiltes 
        self.class_prior_ = self.class_counts / np.sum(self.class_counts)
        self.ftr_log_prob_ =  np.log(self.ftr_counts + 1) - \
            np.log((np.sum(self.ftr_counts, axis=1, keepdims=True) + self.n_classes))

    def predict(self, X_counts):
        # calculate log likelihoods for each class
        log_likelihoods = np.dot(X_counts, self.ftr_log_prob_.T) + np.log(self.class_prior_)
        # predict the class with the highest log likelihood
        return self.classes[np.argmax(log_likelihoods, axis=1)]

class BernulliNBC(object):
    def __init__(self):
        self.class_prior_ = None # (n_classes,)
        self.ftr_log_prob_ = None 
        self.classes = None # (n_classes,)
        self.ftr_counts = None # (n_classes, n_features)
        self.n_classes = None 

    def fit(self, X_bin, y):
        self.classes = np.unique(y) # get unique classes
        self.n_classes = len(self.classes) # num of classes

        self.class_counts = np.zeros(self.n_classes)
        self.ftr_counts = np.zeros((self.n_classes, X_bin.shape[1]))

        for i, cls_ in enumerate(self.classes):
            cls_indices = np.where(y == cls_)[0] # get indices of class cls_
            self.class_counts[i] = len(cls_indices)
            self.ftr_counts[i] = np.array(X_bin[cls_indices].sum(axis=0)).flatten()
        # calculate class priors and feature log probabiltes
        self.class_prior_ = self.class_counts / np.sum(self.class_counts)
        self.ftr_log_prob_ =  np.log(self.ftr_counts + 1) - \
            np.log((np.sum(self.ftr_counts, axis=1, keepdims=True) + 2))

    def predict(self, X_bin):
        log_likelihoods = np.dot(X_bin, self.ftr_log_prob_.T) + np.log(self.class_prior_)
        return self.classes[np.argmax(log_likelihoods, axis=1)]
    

class GaussianNBC(object):
    def __init__(self, eps=1e-6):
        self.classes = None # (n_classes,)
        self.n_classes = None # num of classes
        self.parameters = {
            "mean": None,  # shape: (K, M)
            "sigma": None,  # shape: (K, M)
            "prior": None,  # shape: (K,)
        }
        self.hyperparameters = {"eps": eps}

    def fit(self, X, y):
        """
        Fit the model parameters via maximum likelihood.

        Notes
        -----
        The model parameters are stored in the :py:attr:`parameters
        <numpy_ml.linear_models.GaussianNBClassifier.parameters>` attribute.
        The following keys are present:

            "mean": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature means for each of the `K` label classes
            "sigma": :py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
                Feature variances for each of the `K` label classes
            "prior": :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
                Prior probability of each of the `K` label classes, estimated
                empirically from the training data

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`
        y: :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The class label for each of the `N` examples in `X`

        Returns
        -------
        self : :class:`GaussianNBClassifier <numpy_ml.linear_models.GaussianNBClassifier>` instance   
        """
        self.classes = np.unique(y) # get unique classes
        self.n_classes = len(self.classes) # num of classes
        K = self.n_classes
        N, M = X.shape

        P = self.parameters
        H = self.hyperparameters
        P["mean"] = np.zeros((K, M))
        P["sigma"] = np.zeros((K, M))
        P["prior"] = np.zeros((K,))

        for i, cls_ in enumerate(self.classes):
            cls_indices = np.where(y == cls_)[0] # get indices of class cls_
            X_cls = X[cls_indices]

            P["mean"][i, :] = np.mean(X_cls, axis=0)
            P["sigma"][i, :] = np.var(X_cls, axis=0) + H["eps"]
            P["prior"][i] = len(cls_indices) / N
        return self
    
    def predict(self, X):
        """
        predict the class label for each example in **X** via trained Gaussion NBC.

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`

        Returns
        -------
        labels : :py:class:`ndarray <numpy.ndarray>` of shape `(N)`
            The predicted class labels for each example in `X`
        """
        return self.classes[np.argmax(self._log_posterior(X), axis=1)]

    def _log_posterior(self, X):
        """
        Returns
        -------
        log_posterior : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            Unnormalized log posterior probability of each class for each
            example in `X`

        Notes
        -----
        Unnormalized log posterior for example :math:`\mathbf{x}_i` 
        and class: math:`c` is::

        .. math::

            \log P(y_i = c \mid \mathbf{x}_i, \theta)
                &\propto \log P(y=c \mid \theta) +
                    \log P(\mathbf{x}_i \mid y_i = c, \theta) \\
                &\propto \log P(y=c \mid \theta)
                    \sum{j=1}^M \log P(x_j \mid y_i = c, \theta)

        In the Gaussian naive Bayes model, the feature likelihood for class
        :math:`c`, :math:`P(\mathbf{x}_i \mid y_i = c, \theta)` is assumed to
        be normally distributed

        .. math::

            \mathbf{x}_i \mid y_i = c, \theta \sim \mathcal{N}(\mu_c, \Sigma_c)
        """
        K = self.n_classes
        P = self.parameters
        log_posterior = np.zeros((X.shape[0], K))
        for i in range(K):
            mu = P["mean"][i]
            sigsq = P["sigma"][i]
            prior = P["prior"][i]

            # log likelihood = log X | N(mu, sigsq)
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigsq))
            log_likelihood -= 0.5 * np.sum(((X - mu) ** 2) / sigsq, axis=1)
            log_posterior[:, i] = log_likelihood + np.log(prior)
        return log_posterior
    