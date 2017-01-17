import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
import lxml.etree as et
import numpy as np
import matplotlib
import re
import itertools

matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from features import Featurizer
from learning import LogReg, odds_to_prob, SciKitLR
from learning_mn import MnLogReg

from lstm import *
from learning_utils import test_scores, calibration_plots, training_set_summary_stats, sparse_abs, LF_coverage, \
    LF_overlaps, LF_conflicts, LF_accuracies
from pandas import Series, DataFrame
from random import random
import logging

import math
import numpy as np
from multiprocessing import Process, Queue
from itertools import combinations
import itertools
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix

import networkx as nx
import codecs
import cPickle as pickle
from tokenizer import *

logger = logging.getLogger('snorkel')


class ProgressBar(object):
    def __init__(self, steps):
        self.steps = steps

    def update(self, i):
        progress = math.ceil(i / float(len(self.steps)) * 100)
        sys.stdout.write('Processing \r{:2.2f}% {}/{}'.format(progress, i, self.steps))
        sys.stdout.flush()

    def stop(self):
        sys.stdout.write('Processing \r{:2.2f}% {}/{}'.format(100.0, self.steps, self.steps))
        sys.stdout.flush()
        sys.stdout.write('Complete\n')


def mp_apply_lfs(lfs, candidates, nprocs):
    '''http://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/'''

    def worker(pid, idxs, out_queue):
        # print "\tLF process_id={} {} items".format(pid, len(idxs))
        outdict = {}
        for i in idxs:
            outdict[i] = [lf(candidates[i]) for lf in lfs]
        out_queue.put(outdict)

    out_queue = Queue()
    chunksize = int(math.ceil(len(candidates) / float(nprocs)))
    procs = []

    nums = range(0, len(candidates))
    for i in range(nprocs):
        p = Process(
            target=worker,
            args=(i, nums[chunksize * i:chunksize * (i + 1)],
                  out_queue))
        procs.append(p)
        p.start()

    # Collect all results
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_queue.get())

    for p in procs:
        p.join()

    X = sparse.lil_matrix((len(candidates), len(lfs)))
    for i in sorted(resultdict):
        for j, v in enumerate(resultdict[i]):
            if v != 0:
                X[i, j] = v

    return X.tocsr()


class TrainingSet(object):
    """
    Wrapper data object which applies the LFs to the candidates comprising the training set,
    featurizes them, and then stores the resulting _noisy training set_, as well as the LFs and featurizer.

    As input takes:
        - A set of Candidate objects comprising the training set
        - A set of labeling functions (LFs) which are functions f : Candidate -> {-1,0,1}
        - A Featurizer object, which is applied to the Candidate objects to generate features
    """

    def __init__(self, training_candidates, lfs, featurizer=None, num_procs=1):
        self.num_procs = num_procs
        self.training_candidates = training_candidates
        self.featurizer = featurizer
        self.lfs = lfs
        self.lf_names = [lf.__name__ for lf in lfs]
        self.L, self.F = self.transform(self.training_candidates, fit=True)
        self.dev_candidates = None
        self.dev_labels = None
        self.L_dev = None

        self.summary_stats()

    def transform(self, candidates, fit=False):
        """Apply LFs and featurize the candidates"""
        print "Applying LFs..."
        L = self._apply_lfs(candidates)
        F = None
        if self.featurizer is not None:
            print "Featurizing..."
            F = self.featurizer.fit_transform(candidates) if fit else self.featurizer.transform(candidates)
        return L, F

    def _apply_lfs(self, candidates):
        """Apply the labeling functions to the candidates to populate X"""
        if self.num_procs > 1:
            return mp_apply_lfs(self.lfs, candidates, self.num_procs)
        else:
            X = sparse.lil_matrix((len(candidates), len(self.lfs)))
            for i, c in enumerate(candidates):
                for j, lf in enumerate(self.lfs):
                    X[i, j] = lf(c)
            return X.tocsr()

    def summary_stats(self, return_vals=False, verbose=True):
        """Print out basic stats about the LFs wrt the training candidates"""
        return training_set_summary_stats(self.L, return_vals=return_vals, verbose=verbose)

    def lf_stats(self, dev_candidates=None, dev_labels=None):
        """Returns a pandas Dataframe with the LFs and various per-LF statistics"""
        N, M = self.L.shape

        # Default LF stats
        d = {
            'j': range(len(self.lfs)),
            'coverage': Series(data=LF_coverage(self.L), index=self.lf_names),
            'overlaps': Series(data=LF_overlaps(self.L), index=self.lf_names),
            'conflicts': Series(data=LF_conflicts(self.L), index=self.lf_names)
        }

        # Empirical stats, based on supplied development set
        if dev_candidates and dev_labels is not None:
            if self.L_dev is None or dev_candidates != self.dev_candidates or any(dev_labels != self.dev_labels):
                self.dev_candidates = dev_candidates
                self.dev_labels = dev_labels
                self.L_dev = self._apply_lfs(dev_candidates)
            d['accuracy'] = Series(data=LF_accuracies(self.L_dev, self.dev_labels), index=self.lf_names)
            # counts
            n = sparse_abs(self.L_dev).sum(axis=0)
            n = np.ravel(n).astype(np.int32)
            d["n"] = Series(data=n, index=self.lf_names)

        return DataFrame(data=d, index=self.lf_names)


class Learner(object):
    """
    Core learning class for Snorkel, encapsulating the overall process of learning a generative model of the
    training data set (specifically: of the LF-emitted labels and the true class labels), and then using this
    to train a given noise-aware discriminative model.

    As input takes a TrainingSet object and a NoiseAwareModel object (the discriminative model to train).
    """

    def __init__(self, training_set, model=None, gen_model=LogReg()):
        self.training_set = training_set
        self.model = model

        # We need to know certain properties _that are set in the model defn_
        self.bias_term = self.model.bias_term if hasattr(self.model, 'bias_term') else False

        # Derived objects from the training set
        self.L_train = self.training_set.L
        self.F_train = self.training_set.F
        self.X_train = None
        self.n_train, self.m = self.L_train.shape
        self.f = self.F_train.shape[1] if self.F_train is not None else None

        # Cache the transformed test set as well
        self.test_candidates = None
        self.gold_labels = None
        self.L_test = None
        self.F_test = None
        self.X_test = None

        self.gen_model = gen_model
        self.training_marginals = None

    def _set_model_X(self, L, F):
        """Given LF matrix L, feature matrix F, return the matrix used by the end discriminative model."""
        n, m = L.shape
        X = sparse.hstack([L, F], format='csr')
        if self.bias_term:
            X = sparse.hstack([X, np.ones((n, 1))], format='csr')
        return X

    def train(self, lf_w0=5.0, feat_w0=0.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # Set the initial weights for LFs and feats
        w0 = np.concatenate([lf_w0 * np.ones(self.m), feat_w0 * np.ones(self.f)])
        w0 = np.append(w0, 0) if self.bias_term else w0

        # Construct matrix X for "joint" approach
        self.X_train = self._set_model_X(self.L_train, self.F_train)

        # Train model
        self.model.train(self.X_train, w0=w0, **model_hyperparams)

    def test(self, test_candidates, gold_labels, display=True, return_vals=False, thresh=0.5):
        """
        Apply the LFs and featurize the test candidates, using the same transformation as in training set;
        then test against gold labels using trained model.
        """
        # Cache transformed test set
        if self.X_test is None or test_candidates != self.test_candidates or any(gold_labels != self.gold_labels):
            self.test_candidates = test_candidates
            self.gold_labels = gold_labels
            self.L_test, self.F_test = self.training_set.transform(test_candidates)
            self.X_test = self._set_model_X(self.L_test, self.F_test)

        if display:
            calibration_plots(self.model.marginals(self.X_train), self.model.marginals(self.X_test), gold_labels)

        return test_scores(self.model.predict(self.X_test, thresh=thresh), gold_labels, return_vals=return_vals,
                           verbose=display)

    def lf_weights(self):
        return self.model.w[:self.m]

    def lf_accs(self):
        return odds_to_prob(self.lf_weights())

    def feature_weights(self):
        return self.model.w[self.m:self.m + self.f]

    def predictions(self, thresh=0.5):
        return self.model.predict(self.X_test, thresh=thresh)

    def marginals(self):
        return self.model.marginals(self.X_test)

    def test_mv(self, test_candidates, gold_labels, display=True, return_vals=False):
        """Test *unweighted* majority vote of *just the LFs*"""
        # Ensure that L_test is initialized
        self.test(test_candidates, gold_labels, display=False)

        # L_test * 1
        mv_pred = np.ravel(np.sign(self.L_test.sum(axis=1)))
        return test_scores(mv_pred, gold_labels, return_vals=return_vals, verbose=display)

    def test_wmv(self, test_candidates, gold_labels, display=True, return_vals=False):
        """Test *weighted* majority vote of *just the LFs*"""
        # Ensure that L_test is initialized
        self.test(test_candidates, gold_labels, display=False)

        # L_test * w_lfs
        wmv_pred = np.sign(self.L_test.dot(self.lf_weights()))
        return test_scores(wmv_pred, gold_labels, return_vals=return_vals, verbose=display)

    def feature_stats(self, n_max=100):
        """Return a DataFrame of highest (abs)-weighted features"""
        idxs = np.argsort(np.abs(self.feature_weights()))[::-1][:n_max]
        d = {'j': idxs, 'w': [self.feature_weights()[i] for i in idxs]}
        return DataFrame(data=d, index=[self.training_set.featurizer.feat_inv_index[i] for i in idxs])


class PipelinedLearner(Learner):
    """
    Implements the **"pipelined" approach**- this is the method more literally corresponding
    to the Data Programming paper
    """

    def train_gen_model(self, lf_w0=5.0, **model_hyperparams):
        self.training_marginals = self.train_lf_model(w0=lf_w0, **model_hyperparams)

    def train_disc_model(self, feat_w0=0.0, lf_w0=5.0, **model_hyperparams):
        if self.model:
            self.train_model(self.training_marginals, w0=feat_w0, **model_hyperparams)

    def _set_model_X(self, L, F):
        n, f = F.shape
        X = F.tocsr()
        if self.bias_term:
            X = sparse.hstack([X, np.ones((n, 1))], format='csr')
        return X

    def train_lf_model(self, w0=1.0, **model_hyperparams):
        """Train the first _generative_ model of the LFs"""
        #gen_n_iter = 4000
        #w0 = w0 * np.ones(self.m)
        #self.training_model = self.gen_model
        #self.training_model.train(self.L_train, w0=w0, **model_hyperparams)

        w0 = w0*np.ones(self.m)
        self.training_model =  self.gen_model

        print model_hyperparams
        if "new_encoding" in model_hyperparams:
            self.training_model.train(self.L_train, w0=w0)
        else:
            self.training_model.train(self.L_train, w0=w0, **model_hyperparams)


        # Compute marginal probabilities over the candidates from this model of the training set
        return self.training_model.marginals(self.L_train)

    def train_model(self, training_marginals, w0=0.0, **model_hyperparams):
        """Train the provided end _discriminative_ model"""
        w0 = w0 * np.ones(self.f)
        w0 = np.append(w0, 0) if self.bias_term else w0
        self.X_train = self._set_model_X(self.L_train, self.F_train)
        self.w = self.model.train(self.X_train, training_marginals=training_marginals, \
                                  w0=w0, **model_hyperparams)

    def train(self, feat_w0=0.0, lf_w0=5.0, class_balance=False, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""

        training_marginals = self.train_lf_model(w0=lf_w0, **model_hyperparams)
        self.training_marginals = training_marginals

        # Find the larger class, then subsample, setting the ones we want to drop to 0.5
        if class_balance:
            pos = np.where(training_marginals > 0.5)[0]
            pp = len(pos)
            neg = np.where(training_marginals < 0.5)[0]
            nn = len(neg)
            print "Number of positive:", pp
            print "Number of negative:", nn
            majority = neg if nn > pp else pos
            minority = pos if nn > pp else neg

            # Just set the non-subsampled members to 0.5, so they will be filtered out
            for i in majority:
                if random() > pp / float(nn):
                    training_marginals[i] = 0.5

        print "Training model..."
        self.train_model(training_marginals, w0=feat_w0, **model_hyperparams)

    def lf_weights(self):
        return self.training_model.w

    def feature_weights(self):
        return self.model.w


class RepresentationLearner(PipelinedLearner):
    """
    Implements the _pipelined_ approach for an end model that also learns a representation
    """

    def train_model(self, training_marginals, w0=0.0, **model_hyperparams):
        """Train the provided end _discriminative_ model"""
        self.w = self.model.train(self.training_set.training_candidates, training_marginals=training_marginals, \
                                  **model_hyperparams)

    def test(self, test_candidates, gold_labels, display=True, return_vals=False):
        # Cache transformed test set
        if test_candidates != self.test_candidates or any(gold_labels != self.gold_labels):
            self.test_candidates = test_candidates
            self.gold_labels = gold_labels
        if display:
            calibration_plots(self.model.marginals(self.training_set.training_candidates), \
                              self.model.marginals(self.test_candidates), gold_labels)
        return test_scores(self.model.predict(self.test_candidates), gold_labels, return_vals=return_vals,
                           verbose=display)

    def predictions(self):
        return self.model.predict(self.test_candidates)


class SpanLearner(PipelinedLearner):
    # def __init__(self,split_chars=[]):
    #    self.split_chars=split_chars
    #    pass

    def train_gen_model(self, lf_w0=5.0, **model_hyperparams):
        print "\nTraining generative model..."
        self.training_marginals = self.train_lf_model(w0=lf_w0, **model_hyperparams)

    def train_disc_model(self, feat_w0=0.0, lf_w0=5.0, **model_hyperparams):
        if self.model:
            print "\nTraining discriminitive model..."
            self.train_model(self.training_marginals, w0=feat_w0, **model_hyperparams)

    def _get_sentence_bags(self, candidates):
        '''
        All disjoint (non-overlapping) spans in candidate set

        :param candidates:
        :return:
        '''
        candidates = zip(*sorted([(c.char_start, c) for c in candidates]))[1]

        spans = []
        curr = [candidates[0]]
        for i in range(1, len(candidates)):
            if not curr:
                curr += [candidates[i]]
                continue

            span_overlap = False
            for c in curr:
                if overlaps(c, candidates[i]):
                    span_overlap = True

            if span_overlap:
                curr += [candidates[i]]
            else:
                spans += [curr]
                curr = [candidates[i]]
        if curr:
            spans += [curr]

        for s in spans:
            yield s


    # ===============================================================

    def _overlapping_span(self, a, b):
        return len(set(a).intersection(set(b))) > 0


    def _contained_span(self, a, b):
        v = min(a) >= min(b) and max(a) <= max(b)
        v |= min(b) >= min(a) and max(b) <= max(a)
        return v


    def _get_span_width(self, candidates):
        span = list(itertools.chain.from_iterable([c.idxs for c in candidates]))
        return (min(span), max(span))


    def build_spanset_graph(self, candidates, coverage):
        '''
        TODO: implement this in a clean way using networkx
        '''
        G = nx.Graph()
        p_nodes, n_nodes = [], []
        for i in range(len(candidates)):
            if coverage[i] == 1:
                p_nodes.append(i)
            elif coverage[i] == -1:
                n_nodes.append(i)

        # core nodes (all candidates covered by >= 1 positive LFs)
        for i in range(len(p_nodes)):
            G.add_node(p_nodes[i])
            for j in range(i + 1, len(p_nodes)):
                if p_nodes[i] == p_nodes[j]:
                    continue
                s1 = candidates[p_nodes[i]].idxs
                s2 = candidates[p_nodes[j]].idxs
                if self._overlapping_span(s1, s2):
                    G.add_edge(p_nodes[i], p_nodes[j])

        # fully-contained negative nodes
        stack = []
        for i in range(len(n_nodes)):
            for j in range(len(p_nodes)):
                if n_nodes[i] == p_nodes[j]:
                    continue
                s1 = candidates[n_nodes[i]].idxs
                s2 = candidates[p_nodes[j]].idxs

                if self._contained_span(s1, s2):
                    G.add_edge(n_nodes[i], p_nodes[j])

            if n_nodes[i] not in G.nodes():
                stack.append(n_nodes[i])

        # positive spans
        pos_spans, pos_nodes = [], []
        for cc in nx.connected_components(G):
            pos_nodes.append(cc)
            span = self._get_span_width([candidates[i] for i in cc])
            span = range(span[0], span[1] + 1)
            pos_spans += [span]

        # remove negative bridge nodes between positive spans
        s_rm = []
        for i in range(len(stack)):
            w = 0
            for j in range(len(pos_spans)):
                span1 = candidates[stack[i]].idxs
                span2 = pos_spans[j]
                if self._overlapping_span(span1, span2):
                    w += 1
            if w > 1:
                #print "connection", pos_spans, "|", span1
                #G.remove_node(stack[i])
                s_rm.append(stack[i])
        stack = [i for i in stack if i not in s_rm]

        coverage = np.array([0] * len(candidates[0].sentence["words"]))
        for span in pos_spans:
            coverage[np.array(span)] = 1

        # add candidates sorted by length. greedily add candidates
        # that do not connect with existing candidate spans
        rm = []
        stack = sorted([(i, candidates[i].idxs) for i in stack], key=lambda x: len(x[-1]), reverse=1)
        for node_idx, span in stack:
            if sum(coverage[min(span):max(span) + 1]) == 0:
                G.add_node(node_idx)
                coverage[np.array(span)] = 1
            else:
                rm.append((node_idx, candidates[node_idx].idxs))

        # our core spanset
        components, component_nodes = [], []
        for cc in nx.connected_components(G):
            component_nodes.append(cc)
            span = self._get_span_width([candidates[i] for i in cc])
            span = range(span[0], span[1] + 1)
            components += [(tuple(cc), span)]

        # attempt to add any remaining nodes, ordered by overlap with existing
        # spans and max border
        for nodes_i, span_i in rm:
            edges = []
            for nodes_j, span_j in components:
                if self._overlapping_span(span_i, span_j):
                    edges.append(([nodes_i], nodes_j))

            # don't add candidates that connect components
            if len(edges) > 1:
                continue

            for edge in edges:
                for i in edge[0]:
                    for k in edge[1]:
                        if self._overlapping_span(candidates[i].idxs, candidates[k].idxs):
                            G.add_edge(i, k)

        # create final spanset bags
        spansets = list(nx.connected_components(G))
        spansets = [[candidates[i] for i in span] for span in spansets]

        return spansets

    # ===============================================================





    def _get_bags(self, candidates):
        """
        Create candidate spansets for multinomial classification
        :param candidates:
        :return:
        """
        L = self.training_set.L

        print L.shape
        print len(candidates)

        # building bags by starting with positive labeled candidates
        coverage = np.zeros(L.shape[0])
        coverage[(L < 0).nonzero()[0]] = -1
        coverage[(L > 0).nonzero()[0]] = 1

        # sentence candidates
        sent_cands = defaultdict(list)
        for i, c in enumerate(candidates):
            if coverage[i] == 0:
                continue
            sent_cands[c.sent_id].append(c)

        bags = []
        for i, sent_id in enumerate(sent_cands):
            sent_bags = []
            for b in self._get_sentence_bags(sent_cands[sent_id]):
                sent_bags.append(b)

            # TODO: add heuristic to break up very long spans

            #for b in sent_bags:
            #    print b
            #break


            # sanity check for disjoint bags
            if disjoint(sent_bags):
                print>> sys.stderr, sent_id, "ERROR -- Bags are NOT disjoint"
                disjoint(sent_bags, verbose=True)

            bags += sent_bags

        return bags

    def generate_span_bags(self, thresh=0.0, method="simple",
                           break_on=["/"],
                           **model_hyperparams):
        if method == "simple":
            self.span_bags = self._get_bags(self.training_set.training_candidates)

        elif method == "graph":

            L = self.training_set.L

            candidates = self.training_set.training_candidates

            # building bags by starting with positive labeled candidates
            coverage = np.zeros(L.shape[0])
            coverage[(L < 0).nonzero()[0]] = -1
            coverage[(L > 0).nonzero()[0]] = 1

            # sentence candidates
            sent_cands = defaultdict(list)
            for i, c in enumerate(candidates):
                if coverage[i] == 0:
                    continue
                sent_cands[c.sent_id].append(c)

            ridx = {c:i for i,c in enumerate(candidates)}

            spanbags = []
            for sent_id in sent_cands:
                labels = [coverage[ridx[c]] for c in sent_cands[sent_id] ]
                spanbags.extend( self.build_spanset_graph(sent_cands[sent_id],labels) )
            self.span_bags = spanbags


    def _set_span(self, candidates):
        '''
        Get candidate set span

        :param candidates:
        :return:
        '''
        offsets = list(itertools.chain.from_iterable([[c.char_start, c.char_end] for c in candidates]))
        i, j = min(offsets), max(offsets)
        text = candidates[0].sentence["text"]
        offset = candidates[0].sentence["char_offsets"][0]
        s = text[i - offset:j - offset + 1]
        return s, (i - offset, j - offset + 1)


    # TODO: REFACTOR
    # ---------------------------------

    def train_lf_model(self, w0=5.0, **model_hyperparams):
        '''

        :param w0:
        :param model_hyperparams:
        :return:
        '''

        num_lfs = self.training_set.L.shape[1]
        lf_w = np.array([w0] * num_lfs)

        # multinomial
        if type(self.gen_model) is MnLogReg:

            missing = 0
            self.generate_span_bags()

            Xs, Ys, classes, f_bags, f_instances = self.lf_prob(include_neg=True)

            self.Xs = Xs
            self.Ys = Ys
            self.classes = classes
            self.f_bags = f_bags
            self.f_instances = f_instances

            self.gen_model.train(Xs, w0=lf_w, **model_hyperparams)

            marginals = []
            mn_marginals = self.gen_model.marginals(self.Xs)
            # cand_index = {c: i for i in range(len(f_bags)) for c in f_bags[i]}
            cand_span_index = {}
            for i in range(len(f_bags)):
                for c in f_bags[i]:
                    cand_span_index[c] = i

            # Marginals for multinomials are per spanset, so we need to flatten
            # matrices back to our original observed candidate set candidate
            # Candidates with no LF coverage are assumed 0.5
            for c in self.training_set.training_candidates:

                if c not in cand_span_index:
                    marginals.append(0.5)
                    missing += 1
                    continue

                i = cand_span_index[c]

                if c in self.f_bags[i]:
                    idx = self.f_bags[i].index(c)
                    k = self.f_instances[i][idx]
                    marginals.append(mn_marginals[i][k])
                else:
                    print>>sys.err, "Multinomial Seq Error"
                    marginals.append(0.5)


            self._marginals = mn_marginals

            if missing > 0:
                print>>sys.stderr, "Skipped {}/{} candidates with no LF coverage".format(missing,len(self.training_set.training_candidates))

            return np.array(marginals)

        # binary
        else:
            print "\nBinary generative model"
            w0 = w0 * np.ones(self.m)
            self.gen_model.train(self.L_train, w0=lf_w, **model_hyperparams)
            return self.gen_model.marginals(self.L_train)


    def bag_lf_prob(self, bag, L, sparsity_threshold=1024):
        '''
        Create the M x D_i (num_lfs X num_classes) matrix

        :param bag:
        :param L:
        :param sparsity_threshold:
        :return:
        '''
        num_lfs = self.training_set.L.shape[1]
        candidates = [c for c in bag if c != None]

        # generate class space and candidate instances
        seq, instances = self.candidate_multinomials(candidates)
        P = np.zeros((num_lfs, 2 ** len(seq)))  # M x D_i probabilty distrib

        # HACK
        #if 2 ** len(seq) >= sparsity_threshold:
        #    P = lil_matrix((num_lfs, 2 ** len(seq) ), dtype=np.float32)

        idxs = range(0, len(seq))
        classes = sum([map(list, combinations(idxs, i)) for i in range(len(idxs) + 1)], [])

        if len(instances) != len({s: 1 for s in instances}):
            print "ERROR!!"

        classes = {cls: i for i, cls in enumerate(sorted([tuple(x) for x in classes]))}

        # HEURISTIC: if all LFs vote negative, assign all mass to <N/A>
        assign_na = False
        if np.all(L.toarray() <= 0):
            assign_na = True

        # compute probability matrix
        row, col = L.nonzero()
        for c_i in row:
            for lf_j in col:
                if not assign_na:
                    try:
                        k = classes[instances[c_i]]
                        P[lf_j, k] += L[c_i, lf_j]
                    except:
                        print "FAILURE", k, classes, c_i, P.shape, (lf_j, k)

                else:
                    k = classes[tuple()]
                    P[lf_j, k] += 1

        P = (P.T - np.min(P, axis=1)).T
        P = (P.T / np.sum(P, axis=1)).T
        P[np.isnan(P)] = 0

        class_names = []
        classes = zip(*sorted(classes.items(), key=lambda x: x[1], reverse=0))[0]
        seq = np.array(seq)
        for key in classes:
            if len(key) == 0:
                class_names += ["<N/A>"]
                continue
            class_names += [" ".join(seq[np.array(key)])]

        # HACK -- use sparse format for large P matrices
        if (2 ** len(seq)) > sparsity_threshold:
            P = csr_matrix(P)

        # specific instance indices
        instances = [classes.index(idxs) for idxs in instances]

        return P, classes, class_names, instances


    def candidate_multinomials(self, candidates, split_chars=["/", "-", "+"]):
        '''
        Decompose candidate spanset into corresponding sequence instances
        :param candidates:  spanset candidates
        :param split_chars:
        :return:
        '''
        sentence = candidates[0].sentence
        offset = sentence["char_offsets"][0]
        char_offsets = [i - offset for i in sentence["char_offsets"]]

        # candidate span
        span = list(itertools.chain.from_iterable([[c.char_start, c.char_end] for c in candidates]))
        start, end = min(span), max(span)
        idxs = sorted(list(set(itertools.chain.from_iterable([c.idxs for c in candidates]))))

        # current tokenization
        splits = char_offsets[min(idxs):max(idxs) + 1] + [end - offset + 1]
        tokens = []
        for k in range(len(splits) - 1):
            i, j = splits[k], splits[k + 1]
            t = sentence["text"][i:j].strip()
            tokens += [[t, [i, i + len(t)]]]

        # force additional tokenization
        t_tokens = []
        for t in tokens:
            term, span = t
            rgx = r'([{}]+)+'.format("".join(sorted(split_chars)))
            t_term = re.sub(rgx, r' \1 ', term)

            if t_term != term:
                t_spans = []
                curr = span[0]
                for t1 in t_term.split():
                    t_spans += [[t1, [curr, curr + len(t1)]]]
                    curr = curr + len(t1)
                t_tokens.extend(t_spans)
            else:
                t_tokens.append(t)

        # create candidate instances (subsequences)
        instances = []
        tokens, charmap = zip(*t_tokens)
        for c in candidates:
            offset = c.sentence["char_offsets"][0]
            i, j = c.char_start - offset, c.char_end - offset
            idxs = []
            for k in range(len(charmap)):
                ii, jj = charmap[k]
                jj -= 1  # use token slots, otherwise overlap detection is off by one
                if max(i, ii) <= min(j, jj):
                    idxs += [k]
            instances += [tuple(idxs)]

        return tokens, instances

    def lf_prob(self, include_neg=True, class_threshold=8192):
        '''
        For each spanset, generate a matrix of LF probabilities

        :param include_neg:     include spansets that only include negative LF weights
        :param class_threshold:
        :return:
        '''

        Xs, Ys, strs, f_bags, f_instances = [], [], [], [], []
        L = self.training_set.L
        cand_idx = {c: i for i, c in enumerate(self.training_set.training_candidates)}

        for i, bag in enumerate(self.span_bags):

            # Filer pathologically long cases
            seq, instances = self.candidate_multinomials(bag)
            k = 2 ** len(seq)
            if k > class_threshold:
                print "SKIPPING", class_threshold, k, seq
                continue

            # build LF matrix for bag candidates
            idxs = [cand_idx[c] for c in bag]
            prob, classes, seqs, instances = self.bag_lf_prob(bag, L[idxs])

            if not include_neg and np.all(L[idxs].toarray() <= 0):
                continue

            Xs += [prob]
            Ys += [classes]
            strs += [seqs]
            f_bags += [bag]
            f_instances += [instances]

        return Xs, Ys, strs, f_bags, f_instances


    def export(self, outfile, marginals=None, num_samples=10, coverage_threshold=0.9,
               tagname="Entity", fmt="pkl", threshold=0.0, min_coverage=1):
        '''
        Export sample sentences and candidates
        :param outfile:
        :param marginals:
        :param num_samples:
        :param tagname:
        :param fmt:
        :param threshold:
        :return:
        '''
        # Multinomial-based sampling
        if type(self.gen_model) is MnLogReg:
            if marginals == None:
                marginals = self._marginals
            sampler = self.mn_sample
        else:
            if marginals == None:
                marginals = self.training_marginals
            sampler = self.bin_sample

        sentences = {}
        ner_tags = defaultdict(list)
        for i, (sentence, cands) in enumerate(sampler(marginals, num_samples=num_samples,
                                                      coverage_threshold=coverage_threshold,
                                                      threshold=threshold, min_coverage=min_coverage)):
            sent, tags = tag_sentence(sentence, cands)
            if sent is None:  # HACK sometimes re-tokenization fails due to char offsets.
                print "ERROR"
                continue
            tags = [t + u"-{}".format(tagname) if t != u"O" else t for t in tags]
            ner_tags[sent.id].append(tags)
            sentences[sent.id] = sent

        # dump sample to file
        if fmt == "pkl" or fmt == "all":
            pkl = {}
            for sent_id in sentences:
                sent = sentences[sent_id]
                pkl[sent_id] = {"sent": sent._asdict(), "tags": ner_tags[sent_id]}
            cPickle.dump(pkl, open("{}.pkl".format(outfile), "w"))

        if fmt == "conll" or fmt == "all":
            with codecs.open("{}.conll".format(outfile), "w", "utf-8") as fp:
                for sent_id in sentences:
                    sentence = sentences[sent_id]
                    for tags in ner_tags[sent_id]:
                        tagged = zip(sentence.words, sentence.poses, tags)
                        for word, pos_tag, ner_tag in tagged:
                            tag = (word, pos_tag, ner_tag)
                            fp.write(" ".join(tag) + u"\n")
                        fp.write(u"\n")

        else:
            print>>sys.stderr,"ERROR - unrecognized export format"


    def bin_sample(self, marginals, num_samples=10, format="conll",
               threshold=0.0, min_coverage=1):
        '''Sample spans using heuristics to normalize across overlapping spans'''
        self.span_bags = self._get_bags(self.training_set.training_candidates)

        cands = list(itertools.chain.from_iterable([bag for bag in self.span_bags]))
        sentences = {c.sent_id: c.sentence for c in cands}
        cand_idx = {c:i for i,c in enumerate(self.training_set.training_candidates)}

        sent_bags_idxs = defaultdict(list)
        for i, sent_id in [(i, bag[0].sent_id) for i, bag in enumerate(self.span_bags)]:
            sent_bags_idxs[sent_id].append(i)

        for progress, sent_id in enumerate(sorted(sentences)):

            try:
                samples = []
                for i in range(num_samples):

                    s_cand_set = []
                    for bag_i in sent_bags_idxs[sent_id]:
                        prob = [marginals[cand_idx[c]] for c in self.span_bags[bag_i]]

                        # heuristic to force no sample
                        na = 1.0 - max(prob)
                        prob = [na] + prob
                        cand_set = [None] + self.span_bags[bag_i]
                        prob = [p/sum(prob) for p in prob]

                        # filter probabilities (and remove candidates with p=0.5, i.e., no LF coverage)
                        dist = [[p,c] for p,c in zip(prob,cand_set) if p >= threshold and p != 0.5]
                        m_prob = zip(*dist)[0]
                        prob = [(p / sum(m_prob), c) for p,c in dist]

                        prob, classes = zip(*prob)
                        rs = np.random.choice(classes, 1, p=prob)[0]

                        if rs != None:
                            s_cand_set += [(rs, bag_i)]

                    samples += [s_cand_set]

                # Generate Sentence Samples
                # consisting of sentence + candidate list
                for cs in samples:
                    cs = [c[0] for c in cs]
                    yield (sentences[sent_id], cs)


            except Exception as e:
                print>> sys.stderr, "Warning -- sampling error!", e
                continue


    # --------------
    def predictions(self,threshold=0.5):
        return None


    # --------------



    def mn_sample(self, marginals, num_samples=10, format="conll",
               threshold=0.0, min_coverage=-1, coverage_threshold=0.9):
        '''
        Generate sample sentences using multinomial marginals
        :param marginals:
        :param num_samples:
        :param format:
        :param threshold:
        :return:
        '''
        cands = list(itertools.chain.from_iterable([bag for bag in self.f_bags]))
        sentences = {c.sent_id: c.sentence for c in cands}

        # --------------------------
        # compute token coverage for each sentence, i.e. the ratio of covered/candidate tokens
        # drop sentences where some threshold of token percentage isn't covered
        cand_cover  = {sent_id:np.array([0] * len(sentences[sent_id]["words"])) for sent_id in sentences}
        total_cover = {sent_id: np.array([0] * len(sentences[sent_id]["words"])) for sent_id in sentences}

        for c in self.training_set.training_candidates:
            if c.sent_id not in total_cover:
                continue
            total_cover[c.sent_id][np.array(c.idxs)] = 1

        for c in cands:
            if c.sent_id not in cand_cover:
                continue
            cand_cover[c.sent_id][np.array(c.idxs)] = 1

        tmp = {}
        for sent_id in total_cover:
            if sent_id not in cand_cover:
                print "MISSING", sent_id
            a = cand_cover[sent_id]
            b = total_cover[sent_id]

            coverage = np.logical_and(a,b)
            coverage = coverage.astype(np.int8)

            n = coverage.nonzero()[0].shape[0]
            N = float(b.nonzero()[0].shape[0])

            if (n / N) >= coverage_threshold:
                tmp[sent_id] = sentences[sent_id]

        percent_removed = (1.0 - len(tmp) / float(len(sentences))) * 100
        print "Dropped %.2f%% (%s/%s) sentences at < %.1f%% coverage theshold" % (percent_removed,
                                                                                  (len(sentences)-len(tmp)),
                                                                                  len(sentences),
                                                                                  coverage_threshold*100)
        sentences = tmp
        # --------------------------

        sent_bags_idxs = defaultdict(list)
        for i, sent_id in [(i, bag[0].sent_id) for i, bag in enumerate(self.f_bags)]:
            sent_bags_idxs[sent_id].append(i)

        for progress, sent_id in enumerate(sorted(sentences)):

            try:
                samples = []
                for i in range(num_samples):

                    s_cand_set = []
                    for bag_i in sent_bags_idxs[sent_id]:

                        # restrict samples to known candidates (i.e., ignore impossible samples, like discontinuous spans)
                        prob = sorted(zip(marginals[bag_i], self.classes[bag_i]), reverse=1)
                        mentions = [self.classes[bag_i][j] for j in self.f_instances[bag_i]]

                        # threshold prob. if threshold is too high, choose max(prob)
                        if threshold > 0.0 and max(zip(*prob)[0]) < threshold:
                            threshold = max(zip(*prob)[0])

                        dist = [[p, name] for p, name in prob if
                                (name in mentions or name == '<N/A>') and p >= threshold]
                        m = zip(*dist)[0]
                        p = [(p / sum(m), name) for p, name in dist]
                        p, classes = zip(*p)

                        rs = np.random.choice(classes, 1, p=p)[0]
                        if rs in mentions:
                            rs = self.f_bags[bag_i][mentions.index(rs)]
                        elif rs == "<N/A>":
                            continue
                        else:
                            print>> sys.stderr, "Warning: candidate sample error {}".format(rs)
                            continue

                        s_cand_set += [(rs, bag_i)]

                    samples += [s_cand_set]

                # Generate Sentence Samples
                # consisting of sentence + candidate list
                for cs in samples:
                    cs = [c[0] for c in cs]
                    yield (sentences[sent_id], cs)

            except Exception as e:
               print>> sys.stderr, "Warning -- sampling error!", e
               continue


def overlaps(c1, c2):
    v = c1.doc_id == c2.doc_id
    return v and max(c1.char_start, c2.char_start) <= min(c1.char_end, c2.char_end)

def disjoint(sent_bags, verbose=False):
    ''' SANITY CHECK -- ensure bags are truly disjoint'''
    spans = []
    for bag in sent_bags:
        start, end = 9999999, None
        for c in bag:
            start = min(c.char_start, start)
            end = max(c.char_end, end)
        spans += [(start, end)]

    # overlaps?
    v = False
    for i in range(len(spans)):
        for j in range(len(spans)):
            if i == j:
                continue
            flag = max(spans[i][0], spans[j][0]) <= min(spans[i][1], spans[j][1])
            v |= flag
            if verbose and flag:
                print spans[i], spans[j]
    return v

