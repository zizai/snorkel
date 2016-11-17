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
from lstm import *
from learning_utils import test_scores, calibration_plots, training_set_summary_stats, sparse_abs, LF_coverage, \
    LF_overlaps, LF_conflicts, LF_accuracies
from pandas import Series, DataFrame
from random import random

import math
import numpy as np
from multiprocessing import Process, Queue
from itertools import combinations
from scipy.sparse import csc_matrix,csr_matrix,lil_matrix

import codecs
import cPickle as pickle
from tokenizer import *


def mp_apply_lfs(lfs, candidates, nprocs):
    '''http://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/'''
    def worker(pid, idxs, out_queue):
        #print "\tLF process_id={} {} items".format(pid, len(idxs))
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
    def __init__(self, training_candidates, lfs, featurizer=None, num_procs=1):
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
        self.f               = self.F_train.shape[1] if self.F_train is not None else None

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

    def make_single_bag(self, span_bag, thresh):
        span_bag.append((None, 0.))
        # normalize probability
        # New sampling approach
        for z in range(len(span_bag)):
            span_bag[z] = (span_bag[z][0], span_bag[z][1] if span_bag[z][1] > thresh else 0.)
        s = sum(c[1] for c in span_bag)
        if s > 0.1:
            span_bag = [(c, p / s) for c, p in span_bag]
        else:
            span_bag = None
        return span_bag

    def get_all_sentence(self):
        sentences={}
        for c in self.training_set.training_candidates:
            if c.sent_id not in sentences:
                sentence = c.sentence
                sentence['xmltree'] = None
                sentences[c.sent_id] = sentence
        self.sentences = sentences

    def generate_span_bags(self, thresh=0.0, **model_hyperparams):
        self.train(**model_hyperparams)
        self.get_all_sentence()
        # Group candidates based on sentence id
        candidate_group = dict()
        for c, p in zip(self.training_set.training_candidates, self.training_marginals):
            #if p < thresh: continue
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
                    span_bags.append(self.make_single_bag(span_bag, thresh))
                    span_bag = [i]
                    word_end = i[0].word_end
                else:
                    span_bag.append(i)
                    word_end = max(word_end, i[0].word_end)
            if span_bag != []:
                span_bags.append(self.make_single_bag(span_bag, thresh))
        self.span_bags = [_ for _ in span_bags if _ is not None]

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
                sent_ids = []
                for k, v in span_bag_group_by_sent.iteritems():
                    words = v[0][0][0].sentence['words']
                    poses = v[0][0][0].sentence['poses']
                    sent_ids.append(k)
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
                            # HACK
                            w = w.replace(u"\xa0","_")
                            fp.write(w + ' ' + poses[idx] + ' ' + tags[idx] + '\n')
                        fp.write('\n')
                # sample sentence without tags
                for k, v in self.sentences.iteritems():
                    if k not in sent_ids:
                        for i in range(num_sample):
                            for idx in range(len(v['words'])):
                                fp.write(v['words'][idx] + ' ' + v['poses'][idx] + ' O\n')
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
                sent_ids = []
                for k, v in span_bag_group_by_sent.iteritems():
                    sent_id = v[0][0][0].sent_id
                    cand = v[0][0][0].sentence
                    sent_ids.append(k)
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
                # sample sentence without tags
                for k, v in self.sentences.iteritems():
                    if k not in sent_ids:
                        sent_tags = [['O'] * len(v['poses']) for i in range(num_sample)]
                        output_pkl[k] = {'sent': v, 'tags': sent_tags}
                pickle.dump(output_pkl, open(filename, "w"))
        else:
            print >> sys.stderr, "Unknown output format."
            return


def expand_pos_tag(word, tag):
    pos_tag_map = {"/": ":", "-": ":"}
    tags = [tag if t not in pos_tag_map else pos_tag_map[t] for t in word.split()]
    return tags

def overlaps(c1, c2):
    v = c1.doc_id == c2.doc_id
    return v and max(c1.char_start, c2.char_start) <= min(c1.char_end, c2.char_end)

def align(a, b):
    j = 0
    offsets = []
    for i in range(0, len(a)):
        matched = False
        while not matched and j < len(b):
            if a[i] == b[j]:
                offsets += [(i, j)]
                matched = True
            j += 1
    return offsets

def tokenize(s, split_chars):
    '''Force tokenization'''
    rgx = r'([{}]+)+'.format("".join(split_chars))
    seq = re.sub(rgx, r' \1 ', s)
    seq = seq.replace("'s", " 's")
    return seq.replace("s'", "s '").split()


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


class MultinomialSpanLearner(PipelinedLearner):

    def _get_sentence_bags(self, candidates):
        '''All disjoint (non-overlapping) spans in candidate set'''
        candidates = zip(*sorted([(c.char_start, c) for c in candidates]))[1]

        bags = []
        curr = [candidates[0]]
        for i in range(1,len(candidates)):
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
                bags += [curr]
                curr = [candidates[i]]
        if curr:
            bags += [curr]

        for b in bags:
            yield b


    def _get_bags(self, candidates, split_chars=["/"]):
        """
        Create span bags for multinomial classification
        :param candidates:
        :return:
        """
        L = self.training_set.L
        self.span_classes = {}

        # building bags by starting with positive labeled candidates
        coverage = np.zeros(L.shape[0])
        coverage[(L < 0).nonzero()[0]] = -1
        coverage[(L > 0).nonzero()[0]] = 1

        # sentence candidates
        sent_cands = defaultdict(list)
        for i,c in enumerate(candidates):
            if coverage[i] == 0:
                continue
            sent_cands[c.sent_id].append(c)

        bags = []
        for i, sent_id in enumerate(sent_cands):

            if i % 100 == 0:
                progress = math.ceil(i / float(len(sent_cands)) * 100)
                sys.stdout.write('Processing \r{:2.2f}% {}/{}'.format(progress, i, len(sent_cands)))
                sys.stdout.flush()

            sent_bags = []
            for b in self._get_sentence_bags(sent_cands[sent_id]):
                # create multinomial class names
                k, seq = self._bag_arity(b)
                sent_bags.append(b)

            # TODO: add heuristic to break up very long bags

            # sanity check for disjoint bags
            if disjoint(sent_bags):
                print>>sys.stderr, sent_id, "ERROR -- Bags are NOT disjoint"
                disjoint(sent_bags,verbose=True)

            bags += sent_bags

        print "\nCreated {} bags".format(len(bags))
        return bags


    def train(self, lf_w0=1.0, **model_hyperparams):
        """Train model: **as default, use "joint" approach**"""
        print "Training LF model... {}".format(lf_w0)
        training_marginals = self.train_lf_model(w0=lf_w0, **model_hyperparams)
        self.training_marginals = training_marginals


    def generate_span_bags(self, thresh=0.0, break_on=["/"], **model_hyperparams):
        self.span_bags =  self._get_bags(self.training_set.training_candidates)


    def _bag_arity(self, bag, split_chars=["/", "-"]):
        '''Determine num_words per bag,
        forcing splits on certain characters '''
        s, (i, j) = self._set_span(bag)
        seq = tokenize(s, split_chars)
        return (2 ** len(seq)), seq

    def _set_span(self, candidates):
        offsets = list(itertools.chain.from_iterable([[c.char_start, c.char_end] for c in candidates]))
        i, j = min(offsets), max(offsets)
        text = candidates[0].sentence["text"]
        offset = candidates[0].sentence["char_offsets"][0]
        s = text[i - offset:j - offset + 1]
        return s, (i - offset, j - offset + 1)

    def bag_lf_prob(self, bag, L, sparsity_threshold=1024):
        '''Create the M x D_i (num_lfs X num_classes) matrix
        '''
        num_lfs = self.training_set.L.shape[1]

        k, seq = self._bag_arity(bag)
        candidates = [c for c in bag if c != None]

        # M x D_i probabilty distrib
        P = np.zeros((num_lfs, k))

        # transform candidates into multinomial seqs
        samples = self.candidate_multinomials(candidates)
        idxs = range(0, len(seq))
        classes = sum([map(list, combinations(idxs, i)) for i in range(len(idxs) + 1)], [])
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
                        k = classes[samples[c_i]]
                        P[lf_j, k] += L[c_i, lf_j]
                    except:
                        print "FAILURE", k, classes, c_i, P.shape, (lf_j, k)

                else:
                    k = classes[tuple()]
                    P[lf_j, k] += 1

        P = (P.T - np.min(P, axis=1)).T
        P = (P.T / np.sum(P, axis=1)).T
        P[np.isnan(P)] = 0

        # force sparse matrix give some candidate space size
        if k > sparsity_threshold:
            P = csr_matrix(P)

        strs = []
        classes = zip(*sorted(classes.items(), key=lambda x: x[1], reverse=0))[0]
        seq = np.array(seq)
        for key in classes:
            if len(key) == 0:
                strs += ["<N/A>"]
                continue
            strs += [" ".join(seq[np.array(key)])]

        return P, classes, strs

    def candidate_multinomials(self, candidates, split_chars=["/", "-"]):
        '''convert candidates to tokenized binary sequences '''
        seq, _ = self._set_span(candidates)
        span = tokenize(seq, split_chars)
        classes = []

        # char to word idx
        s, (i, j) = self._set_span(candidates)
        seq = tokenize(s, split_chars)

        for c in candidates:
            mention = tokenize(c.get_attrib_span("words"), split_chars)
            tags = np.array([0] * len(span))
            for i, j in align(span, span):
                tags[j] = 1
            tags = tuple([j for i, j in align(mention, span)])
            classes += [tags]

        return classes

    def lf_prob(self, include_neg=False, class_threshold=1024):

        if "span_bags" not in self.__dict__:
            self.generate_span_bags()

        Xs, Ys, strs, f_bags = [], [], [], []
        L = self.training_set.L

        cand_idx = {c: i for i, c in enumerate(self.training_set.training_candidates)}

        for i, bag in enumerate(self.span_bags):

            if i % 100 == 0:
                progress = math.ceil(i / float(len(self.span_bags)) * 100)
                sys.stdout.write('Processing \r{:2.2f}% {}/{}'.format(progress, i, len(self.span_bags)))
                sys.stdout.flush()

            # skip degenerate long classes
            k, _ = self._bag_arity(bag)
            if k > class_threshold:
                continue

            # build LF matrix for bag candidates
            idxs = [cand_idx[c] for c in bag]
            prob, classes, seqs = self.bag_lf_prob(bag, L[idxs])

            if not include_neg and np.all(L[idxs].toarray() <= 0):
                continue

            Xs += [prob]
            Ys += [classes]
            strs += [seqs]
            f_bags += [bag]

        self.Xs = Xs
        self.Ys = Ys
        self.strs = strs
        self.f_bags = f_bags

        return Xs, Ys, strs, f_bags


    def export(self, marginals, num_samples, outfile,
               tagname="DISEASE", fmt="pkl", threshold=0.0):
        '''Export tagged sentences to file of given format'''
        sentences = {}
        ner_tags = defaultdict(list)
        for i, (sentence, cands) in enumerate(self.sample(marginals, num_samples=num_samples, threshold=threshold)):
            sent, tags = tag_sentence(sentence, cands)
            tags = [t + "-{}".format(tagname) if t != "O" else t for t in tags]
            ner_tags[sent.id].append(tags)
            sentences[sent.id] = sent

        if fmt == "pkl":
            pkl = {}
            for sent_id in sentences:
                sent = sentences[sent_id]
                pkl[sent_id] = {"sent":sent._asdict(),"tags":ner_tags[sent_id]}
            cPickle.dump(pkl, open(outfile,"w"))

        elif fmt == "conll":
            with codecs.open(outfile,"w","utf-8") as fp:
                for sent_id in sentences:
                    sentence = sentences[sent_id]
                    for tags in ner_tags[sent_id]:
                        tagged = zip(sentence.words, sentence.poses, tags)
                        for word, pos_tag, ner_tag in tagged:
                            #ner_tag = ner_tag + "-{}".format(tagname) if ner_tag != "O" else ner_tag
                            tag = (word, pos_tag, ner_tag)
                            fp.write(" ".join(tag) + u"\n")
                        fp.write(u"\n")


    def sample(self, marginals, num_samples=10, format="conll",
               threshold=0.0, show_progress=False):

        cands = list(itertools.chain.from_iterable([bag for bag in self.f_bags]))
        sentences = {c.sent_id:c.sentence for c in cands}

        sent_bags_idxs = defaultdict(list)
        for i, sent_id in [(i, bag[0].sent_id) for i, bag in enumerate(self.f_bags)]:
            sent_bags_idxs[sent_id].append(i)

        for progress, sent_id in enumerate(sorted(sentences)):

            if show_progress and progress % 100 == 0 or progress == len(sentences):
                sys.stdout.write('Sampling \r{:2.2f}% {}/{}'.format( ( progress / float(len(sentences)) * 100),
                                                                     progress, len(sentences) ))
                sys.stdout.flush()

            try:
                # Candidate (bag) Samples
                samples = []
                for i in range(num_samples):

                    s_cand_set = []
                    for bag_i in sent_bags_idxs[sent_id]:

                        # restrict samples to known candidates (i.e., ignore impossible samples, like discontinuous spans)
                        prob = sorted(zip(marginals[bag_i],self.strs[bag_i]),reverse=1)
                        mentions = []
                        for m in [c.get_attrib_span("words") for c in self.f_bags[bag_i]]:
                            mentions += [" ".join(tokenize(m, split_chars=["/", "-"]))]

                        # threshold prob. if threshold is too high, choose max(prob)
                        if threshold > 0.0 and max(zip(*prob)[0]) < threshold:
                            threshold = max(zip(*prob)[0])

                        dist = [[p, name] for p,name in prob if (name in mentions or name == '<N/A>') and p >= threshold]
                        m = zip(*dist)[0]
                        p = [(p/sum(m),name) for p,name in dist]

                        p,classes = zip(*p)
                        rs = np.random.choice(classes, 1, p=p)[0]
                        if rs in mentions:
                            rs = self.f_bags[bag_i][mentions.index(rs)]
                        elif rs == "<N/A>":
                            #rs = None
                            continue
                        else:
                            print>>sys.stderr,"Warning: candidate sample error {}".format(rs)
                            continue
                        s_cand_set += [(rs, bag_i)]

                    samples += [s_cand_set]

                # Generate Sentence Samples
                for cs in samples:
                    cs = [c[0] for c in cs]
                    yield (sentences[sent_id], cs)

            except Exception as e:
                print>>sys.stderr,"Warning -- sampling error!"
                continue
