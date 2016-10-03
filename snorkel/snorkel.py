import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
import lxml.etree as et
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from features import Featurizer
from learning import LogReg, odds_to_prob, SciKitLR
from lstm import *
from learning_utils import test_scores, calibration_plots, training_set_summary_stats, sparse_abs, LF_coverage, \
    LF_overlaps, LF_conflicts, LF_accuracies
from pandas import Series, DataFrame
from random import random

import math
import numpy as np
from multiprocessing import Process, Queue

import codecs
import cPickle as pickle

def mp_apply_lfs(lfs, candidates, nprocs):
    '''http://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/'''
    def worker(pid, idxs, out_queue):
        print "\tLF process_id={} {} items".format(pid, len(idxs))
        outdict = {}
        for i in idxs:
            outdict[i] = [lf(candidates[i]) for lf in lfs]
        out_queue.put(outdict)

    out_queue = Queue()
    chunksize = int(math.ceil(len(candidates) / float(nprocs)))
    procs = []

    nums = range(0,len(candidates))
    for i in range(nprocs):
        p = Process(
                target=worker,
                args=(i,nums[chunksize * i:chunksize * (i + 1)],
                      out_queue))
        procs.append(p)
        p.start()

    # Collect all results 
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_queue.get())

    for p in procs:
        p.join()

    print "Building sparse LF matrix"
    X = sparse.lil_matrix((len(candidates), len(lfs)))
    for i in sorted(resultdict):
        for j,v in enumerate(resultdict[i]):
            if v != 0:
                X[i,j] = v

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
    def __init__(self, training_candidates, lfs, featurizer=None, num_procs=6):
        self.num_procs  = num_procs
        self.training_candidates = training_candidates
        self.featurizer          = featurizer
        self.lfs                 = lfs
        self.lf_names            = [lf.__name__ for lf in lfs]
        self.L, self.F           = self.transform(self.training_candidates, fit=True)
        self.dev_candidates      = None
        self.dev_labels          = None
        self.L_dev               = None
        
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
            for i,c in enumerate(candidates):
                for j,lf in enumerate(self.lfs):
                    X[i,j] = lf(c)
            return X.tocsr()

    def summary_stats(self, return_vals=False, verbose=True):
        """Print out basic stats about the LFs wrt the training candidates"""
        return training_set_summary_stats(self.L, return_vals=return_vals, verbose=verbose)

    def lf_stats(self, dev_candidates=None, dev_labels=None):
        """Returns a pandas Dataframe with the LFs and various per-LF statistics"""
        N, M = self.L.shape

        # Default LF stats
        d = {
            'j'         : range(len(self.lfs)),
            'coverage'  : Series(data=LF_coverage(self.L), index=self.lf_names),
            'overlaps'  : Series(data=LF_overlaps(self.L), index=self.lf_names),
            'conflicts' : Series(data=LF_conflicts(self.L), index=self.lf_names)
        }
        
        # Empirical stats, based on supplied development set
        if dev_candidates and dev_labels is not None:
            if self.L_dev is None or dev_candidates != self.dev_candidates or any(dev_labels != self.dev_labels):
                self.dev_candidates = dev_candidates
                self.dev_labels     = dev_labels
                self.L_dev          = self._apply_lfs(dev_candidates)
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
    def __init__(self, training_set, model=None):
        self.training_set = training_set
        self.model        = model

        # We need to know certain properties _that are set in the model defn_
        self.bias_term = self.model.bias_term if hasattr(self.model, 'bias_term') else False

        # Derived objects from the training set
        self.L_train         = self.training_set.L
        self.F_train         = self.training_set.F
        self.X_train         = None
        self.n_train, self.m = self.L_train.shape
        self.f               = self.F_train.shape[1]

        # Cache the transformed test set as well
        self.test_candidates = None
        self.gold_labels     = None
        self.L_test          = None
        self.F_test          = None
        self.X_test          = None

    def _set_model_X(self, L, F):
        """Given LF matrix L, feature matrix F, return the matrix used by the end discriminative model."""
        n, m = L.shape
        X    = sparse.hstack([L, F], format='csr')
        if self.bias_term:
            X = sparse.hstack([X, np.ones((n, 1))], format='csr')
        return X

    def train(self, lf_w0=5.0, feat_w0=0.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        # Set the initial weights for LFs and feats
        w0 = np.concatenate([lf_w0*np.ones(self.m), feat_w0*np.ones(self.f)])
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
            self.test_candidates     = test_candidates
            self.gold_labels         = gold_labels
            self.L_test, self.F_test = self.training_set.transform(test_candidates)
            self.X_test              = self._set_model_X(self.L_test, self.F_test)
        if display:
            calibration_plots(self.model.marginals(self.X_train), self.model.marginals(self.X_test), gold_labels)
        return test_scores(self.model.predict(self.X_test, thresh=thresh), gold_labels, return_vals=return_vals, verbose=display)

    def lf_weights(self):
        return self.model.w[:self.m]

    def lf_accs(self):
        return odds_to_prob(self.lf_weights())

    def feature_weights(self):
        return self.model.w[self.m:self.m+self.f]
        
    def predictions(self, thresh=0.5):
        return self.model.predict(self.X_test, thresh=thresh)

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
    def _set_model_X(self, L, F):
        n, f = F.shape
        X    = F.tocsr()
        if self.bias_term:
            X = sparse.hstack([X, np.ones((n, 1))], format='csr')
        return X

    def train_lf_model(self, w0=1.0, **model_hyperparams):
        """Train the first _generative_ model of the LFs"""
        w0 = w0*np.ones(self.m)
        self.training_model =  LogReg() #SciKitLR() 
        
        self.training_model.train(self.L_train, w0=w0, **model_hyperparams)

        # Compute marginal probabilities over the candidates from this model of the training set
        return self.training_model.marginals(self.L_train)

    def train_model(self, training_marginals, w0=0.0, **model_hyperparams):
        """Train the provided end _discriminative_ model"""
        w0           = w0*np.ones(self.f)
        w0           = np.append(w0, 0) if self.bias_term else w0
        self.X_train = self._set_model_X(self.L_train, self.F_train)
        self.w       = self.model.train(self.X_train, training_marginals=training_marginals, \
                        w0=w0, **model_hyperparams)

    def train(self, feat_w0=0.0, lf_w0=5.0, class_balance=False, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        print "Training LF model... {}".format(lf_w0)
        training_marginals = self.train_lf_model(w0=lf_w0, **model_hyperparams)
        self.training_marginals = training_marginals

        # Find the larger class, then subsample, setting the ones we want to drop to 0.5
        if class_balance:
            pos = np.where(training_marginals > 0.5)[0]
            pp  = len(pos)
            neg = np.where(training_marginals < 0.5)[0]
            nn  = len(neg)
            print "Number of positive:", pp
            print "Number of negative:", nn
            majority = neg if nn > pp else pos

            # Just set the non-subsampled members to 0.5, so they will be filtered out
            for i in majority:
                if random() > pp/float(nn):
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
            self.gold_labels     = gold_labels
        if display:
            calibration_plots(self.model.marginals(self.training_set.training_candidates), \
                                self.model.marginals(self.test_candidates), gold_labels)
        return test_scores(self.model.predict(self.test_candidates), gold_labels, return_vals=return_vals, verbose=display)

    def predictions(self):
        return self.model.predict(self.test_candidates)

class CRFSpanLearner(PipelinedLearner):
    """
    Implements the _pipelined_ approach for an end model that also learns a representation
    """
    def train(self, lf_w0=5.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        print "Training LF model... {}".format(lf_w0)
        training_marginals = self.train_lf_model(w0=lf_w0, **model_hyperparams)
        self.training_marginals = training_marginals

    def generate_span_bag(self, **model_hyperparams):
        self.train(**model_hyperparams)

        # Group candidates based on sentence id
        candidate_group = dict()
        for c, p in zip(self.training_set.training_candidates, self.training_marginals):
            if c.sent_id not in candidate_group:
                candidate_group[c.sent_id] = []
            candidate_group[c.sent_id].append((c, p))

        span_bags = []
        for k, v in candidate_group.iteritems():
            v.sort(key=lambda x: x[0].word_start, reverse=False)
            span_bag = []
            word_end = -1
            for i in v:
                if word_end != -1 and i[0].word_start > word_end:
                    # N/A class
                    span_bag.append((None, 1. - max(c[1] for c in span_bag)))
                    # normalize probability
                    s = sum(c[1] for c in span_bag)
                    span_bags.append([(c, p / s) for c, p in span_bag])
                    span_bag = []
                    word_end = -1
                else:
                    span_bag.append(i)
                    word_end = i[0].word_end
        self.span_bags = span_bags

    def print_to_file(self, tag = '', num_sample = 10, filename = 'conll_format_data.txt', format = 'conll'):
        if format == 'conll':
            with codecs.open(filename,"w","utf-8") as fp:
                span_bag_group_by_sent = {}
                for span_bag in self.span_bags:
                    sent_id = span_bag[0][0].sent_id
                    if sent_id not in span_bag_group_by_sent:
                        span_bag_group_by_sent[sent_id] = []
                    span_bag_group_by_sent[sent_id].append(span_bag)
                self.span_bag_group_by_sent = span_bag_group_by_sent
                for k, v in span_bag_group_by_sent.iteritems():
                    words = v[0][0][0].sentence['words']
                    samples = []
                    for span_bag in v:
                        samples.append(np.random.choice(len(span_bag), num_sample, p=[c[1] for c in span_bag]))
                    for i in range(num_sample):
                        tags = ['O'] * len(words)
                        for idx, span_bag in enumerate(v):
                            c = span_bag[samples[idx][i]][0]
                            if c is None: continue
                            for x in c.idxs:
                                if x == min(c.idxs):
                                    tags[x] = 'B-' + tag.strip()
                                else:
                                    tags[x] = 'I-' + tag.strip()
                        for idx, w in enumerate(words):
                            fp.write(w + ' ' + tags[idx] + '\n')
                        fp.write('\n')
        elif format == 'pkl':
                span_bag_group_by_sent = {}
                for span_bag in self.span_bags:
                    sent_id = span_bag[0][0].sent_id
                    if sent_id not in span_bag_group_by_sent:
                        span_bag_group_by_sent[sent_id] = []
                    span_bag_group_by_sent[sent_id].append(span_bag)
                self.span_bag_group_by_sent = span_bag_group_by_sent
                output_pkl = dict()
                for k, v in span_bag_group_by_sent.iteritems():
                    sent_id = v[0][0][0].sent_id
                    cand = v[0][0][0].sentence
                    cand['xmltree'] = None
                    sent_tags = []
                    words = v[0][0][0].sentence['words']
                    samples = []
                    for span_bag in v:
                        samples.append(np.random.choice(len(span_bag), num_sample, p=[c[1] for c in span_bag]))
                    for i in range(num_sample):
                        tags = ['O'] * len(words)
                        for idx, span_bag in enumerate(v):
                            c = span_bag[samples[idx][i]][0]
                            if c is None: continue
                            for x in c.idxs:
                                if x == min(c.idxs):
                                    tags[x] = 'B-' + tag.strip()
                                else:
                                    tags[x] = 'I-' + tag.strip()
                        sent_tags.append(tags)
                    output_pkl[sent_id] = {'sent': cand, 'tags': sent_tags}
                pickle.dump(output_pkl, open(filename, "w"))
        else:
            print >> sys.stderr, "Unknown output format."
            return
