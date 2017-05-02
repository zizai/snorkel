from .constants import *
from .disc_learning import NoiseAwareModel
from .utils import MentionScorer
from numbskull import NumbSkull
from numbskull.inference import FACTORS
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
import numpy as np
import random
import scipy.sparse as sparse
from utils import exact_data, log_odds, odds_to_prob, sample_data, sparse_abs, transform_sample_stats


class CoralModel(object):
    def __init__(self, class_prior=False, lf_prior=False, lf_propensity=False, lf_class_propensity=False, seed=271828):
        self.class_prior = class_prior
        self.lf_prior = lf_prior
        self.lf_propensity = lf_propensity
        self.lf_class_propensity = lf_class_propensity
        self.weights = None

        self.rng = random.Random()
        self.rng.seed(seed)

    def train(self, V, cardinality, L, L_offset, y=None, deps=(), init_acc = 1.0, init_deps=1.0, init_class_prior=-1.0, epochs=100, step_size=None, decay=0.99, reg_param=0.1, reg_type=2, verbose=False,
              truncation=10, burn_in=50, timer=None):

        n_data = V.shape[0]
        step_size = step_size or 1.0 / n_data
        reg_param_scaled = reg_param / n_data
        # self._process_dependency_graph(L, deps)
        weight, variable, factor, ftv, domain_mask, n_edges = self._compile(V, cardinality, L, L_offset, y, init_acc) # , init_deps, init_class_prior)

        fg = NumbSkull(n_inference_epoch=0, n_learning_epoch=epochs, stepsize=step_size, decay=decay,
                       reg_param=reg_param_scaled, regularization=reg_type, truncation=truncation,
                       quiet=(not verbose), verbose=verbose, learn_non_evidence=True, burn_in=burn_in)
        fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)

        if timer is not None:
            timer.start()
        fg.learning(out=False)
        if timer is not None:
            timer.end()

        print fg.factorGraphs[0].weight_value
        # self._process_learned_weights(L, fg)

    def marginals(self, L):
        if self.weights is None:
            raise ValueError("Must fit model with train() before computing marginal probabilities.")

        marginals = np.ndarray(L.shape[0], dtype=np.float64)

        for i in range(L.shape[0]):
            logp_true = self.weights.class_prior
            logp_false = -1 * self.weights.class_prior

            l_i = L[i].tocoo()

            for l_index1 in range(l_i.nnz):
                data_j, j = l_i.data[l_index1], l_i.col[l_index1]
                if data_j == 1:
                    logp_true  += self.weights.lf_accuracy_log_odds[j]
                    logp_false -= self.weights.lf_accuracy_log_odds[j]
                    logp_true  += self.weights.lf_class_propensity[j]
                    logp_false -= self.weights.lf_class_propensity[j]
                elif data_j == -1:
                    logp_true  -= self.weights.lf_accuracy_log_odds[j]
                    logp_false += self.weights.lf_accuracy_log_odds[j]
                    logp_true  += self.weights.lf_class_propensity[j]
                    logp_false -= self.weights.lf_class_propensity[j]
                else:
                    ValueError("Illegal value at %d, %d: %d. Must be in {-1, 0, 1}." % (i, j, data_j))

                for l_index2 in range(l_i.nnz):
                    data_k, k = l_i.data[l_index2], l_i.col[l_index2]
                    if j != k:
                        if data_j == -1 and data_k == 1:
                            logp_true += self.weights.dep_fixing[j, k]
                        elif data_j == 1 and data_k == -1:
                            logp_false += self.weights.dep_fixing[j, k]

                        if data_j == 1 and data_k == 1:
                            logp_true += self.weights.dep_reinforcing[j, k]
                        elif data_j == -1 and data_k == -1:
                            logp_false += self.weights.dep_reinforcing[j, k]

            marginals[i] = 1 / (1 + np.exp(logp_false - logp_true))

        return marginals

    def score(self, session, X_test, test_labels, gold_candidate_set=None, b=0.5, set_unlabeled_as_neg=True,
              display=True, scorer=MentionScorer, **kwargs):
        
        # Get the test candidates
        test_candidates = [X_test.get_candidate(session, i) for i in xrange(X_test.shape[0])]

        # Initialize scorer
        s               = scorer(test_candidates, test_labels, gold_candidate_set)
        test_marginals  = self.marginals(X_test, **kwargs)

        return s.score(test_marginals, train_marginals=None, b=b,
                       set_unlabeled_as_neg=set_unlabeled_as_neg, display=display)

    def _process_dependency_graph(self, L, deps):
        """
        Processes an iterable of triples that specify labeling function dependencies.

        The first two elements of the triple are the labeling functions to be modeled as dependent. The labeling
        functions are specified using their column indices in `L`. The third element is the type of dependency.
        Options are :const:`DEP_SIMILAR`, :const:`DEP_FIXING`, :const:`DEP_REINFORCING`, and :const:`DEP_EXCLUSIVE`.

        The results are :class:`scipy.sparse.csr_matrix` objects that represent directed adjacency matrices. They are
        set as various GenerativeModel members, two for each type of dependency, e.g., `dep_similar` and `dep_similar_T`
        (its transpose for efficient inverse lookups).

        :param deps: iterable of tuples of the form (lf_1, lf_2, type)
        """
        dep_name_map = {
            DEP_SIMILAR: 'dep_similar',
            DEP_FIXING: 'dep_fixing',
            DEP_REINFORCING: 'dep_reinforcing',
            DEP_EXCLUSIVE: 'dep_exclusive'
        }

        for dep_name in GenerativeModel.dep_names:
            setattr(self, dep_name, sparse.lil_matrix((L.shape[1], L.shape[1])))

        for lf1, lf2, dep_type in deps:
            if lf1 == lf2:
                raise ValueError("Invalid dependency. Labeling function cannot depend on itself.")

            if dep_type in dep_name_map:
                dep_mat = getattr(self, dep_name_map[dep_type])
            else:
                raise ValueError("Unrecognized dependency type: " + unicode(dep_type))

            dep_mat[lf1, lf2] = 1

        for dep_name in GenerativeModel.dep_names:
            setattr(self, dep_name, getattr(self, dep_name).tocoo(copy=True))

    def _compile(self, V, cardinality, L, L_offset, y, init_acc):
        """
        Compiles a generative model based on L and the current labeling function dependencies.
        """

        # TODO: error checking

        n_data = V.shape[0]
        n_vocab = V.shape[1]
        n_lf = len(L)

        n_weights = n_lf
        n_vars = n_data * (n_vocab + 1)
        n_factors = n_data * n_weights
        n_edges = n_data * (sum([len(l) + 1 for l in L]))

        weight = np.zeros(n_weights, Weight)
        variable = np.zeros(n_vars, Variable)
        factor = np.zeros(n_factors, Factor)
        ftv = np.zeros(n_edges, FactorToVar)
        domain_mask = np.zeros(n_vars, np.bool)

        #
        # Compiles weight matrix
        #
        for i in range(n_weights):
            weight[i]['isFixed'] = False
            weight[i]['initialValue'] = np.float64(init_acc)

        #
        # Compiles variable matrix
        #

        # True Label y
        for i in range(n_data):
            variable[i]['isEvidence'] = False if (y is None) else True
            variable[i]['initialValue'] = self.rng.randrange(0, 2) if (y is None) else (1 if y[i] == 1 else 0)
            variable[i]["dataType"] = 0
            variable[i]["cardinality"] = 2

        # Vocabulary
        for i in range(n_data):
            for j in range(n_vocab):
                variable[n_data + i * n_vocab + j]["isEvidence"] = True
                variable[n_data + i * n_vocab + j]["initialValue"] = V[i, j]
                variable[n_data + i * n_vocab + j]["dataType"] = 0
                variable[n_data + i * n_vocab + j]["cardinality"] = cardinality[j]

        #
        # Compiles factor and ftv matrices
        #
        index = 0
        for i in range(n_data):
            for j in range(n_lf):
                factor[i * n_lf + j]["factorFunction"] = L_offset + j
                factor[i * n_lf + j]["weightId"] = j
                factor[i * n_lf + j]["featureValue"] = 1.0
                factor[i * n_lf + j]["arity"] = len(L[j]) + 1
                factor[i * n_lf + j]["ftv_offset"] = index
                for k in range(len(L[j])):
                    ftv[index]["vid"] = n_data + i * n_vocab + L[j][k]
                    ftv[index]["dense_equal_to"] = 0 # not actually used
                    index += 1
                ftv[index]["vid"] = i
                ftv[index]["dense_equal_to"] = 0 # not actually used
                index += 1

        return weight, variable, factor, ftv, domain_mask, n_edges

    def _compile_output_factors(self, L, factors, factors_offset, ftv, ftv_offset, weight_offset, factor_name, vid_funcs):
        """
        Compiles factors over the outputs of labeling functions, i.e., for which there is one weight per labeling
        function and one factor per labeling function-candidate pair.
        """
        m, n = L.shape

        for i in range(m):
            for j in range(n):
                factors_index = factors_offset + n * i + j
                ftv_index = ftv_offset + len(vid_funcs) * (n * i + j)

                factors[factors_index]["factorFunction"] = FACTORS[factor_name]
                factors[factors_index]["weightId"] = weight_offset + j
                factors[factors_index]["featureValue"] = 1
                factors[factors_index]["arity"] = len(vid_funcs)
                factors[factors_index]["ftv_offset"] = ftv_index

                for i_var, vid_func in enumerate(vid_funcs):
                    ftv[ftv_index + i_var]["vid"] = vid_func(m, n, i, j)

        return factors_offset + m * n, ftv_offset + len(vid_funcs) * m * n, weight_offset + n

    def _compile_dep_factors(self, L, factors, factors_offset, ftv, ftv_offset, weight_offset, j, k, factor_name, vid_funcs):
        """
        Compiles factors for dependencies between pairs of labeling functions (possibly also depending on the latent
        class label).
        """
        m, n = L.shape

        for i in range(m):
            factors_index = factors_offset + i
            ftv_index = ftv_offset + len(vid_funcs) * i

            factors[factors_index]["factorFunction"] = FACTORS[factor_name]
            factors[factors_index]["weightId"] = weight_offset
            factors[factors_index]["featureValue"] = 1
            factors[factors_index]["arity"] = len(vid_funcs)
            factors[factors_index]["ftv_offset"] = ftv_index

            for i_var, vid_func in enumerate(vid_funcs):
                ftv[ftv_index + i_var]["vid"] = vid_func(m, n, i, j, k)

        return factors_offset + m, ftv_offset + len(vid_funcs) * m, weight_offset + 1

    def _process_learned_weights(self, L, fg):
        _, n = L.shape

        w = fg.getFactorGraph().getWeights()
        weights = GenerativeModelWeights(n)

        if self.class_prior:
            weights.class_prior = w[0]
            w_off = 1
        else:
            w_off = 0

        weights.lf_accuracy_log_odds = np.copy(w[w_off:w_off + n])
        w_off += n

        for optional_name in GenerativeModel.optional_names:
            if getattr(self, optional_name):
                setattr(weights, optional_name, np.copy(w[w_off:w_off + n]))
                w_off += n

        for dep_name in self.dep_names:
            mat = getattr(self, dep_name)
            weight_mat = sparse.lil_matrix((n, n))

            for i in range(len(mat.data)):
                if w[w_off] != 0:
                    weight_mat[mat.row[i], mat.col[i]] = w[w_off]
                w_off += 1

            setattr(weights, dep_name, weight_mat.tocsr(copy=True))

        self.weights = weights
