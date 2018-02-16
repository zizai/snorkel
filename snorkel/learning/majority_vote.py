import numpy as np

from classifier import Classifier

class MajorityVoter(Classifier):
    """Simple Classifier that makes the majority vote given an AnnotationMatrix."""

    def marginals(self, X, **kwargs):
        return np.where(np.ravel(np.sum(X, axis=1)) <= 0, 0.0, 1.0)

class SoftMajorityVoter(Classifier):
    """Simple Classifier that makes the 'soft' majority vote given an AnnotationMatrix."""

    def marginals(self, X, **kwargs):
        net_votes = np.sum(X, axis=1)
        num_votes = np.sum(abs(X), axis=1)
        marginals = np.ravel(np.divide(net_votes + num_votes, 2.0 * num_votes))
        marginals[np.where(np.isnan(marginals))] = 0.5
        return marginals