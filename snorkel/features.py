import os, sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
import itertools
from pandas import DataFrame

# Feature modules
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from utils import get_as_dict
from entity_features import *

import string
import fuzzy
import pyphen
#pyphen.language_fallback('nl_NL_variant1')
morphology = pyphen.Pyphen(lang="en_Latn_US") #pyphen.Pyphen(lang='en_US')

soundex = fuzzy.Soundex(4)


def letter_ratio(c,idxs,bins=20):
    s = c.get_attrib_span("words")
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
    c = float(sum([1 for ch in s if ch in punc]))
    w = (1.0 - (c/float(len(s)))) * 100
    w = int(w)
    return "LETTER_RATIO_[{}]".format( int(w/float(bins)) )
    
    
def vowel_ratio(c,idxs,bins=20):
    s = c.get_attrib_span("words")
    punc = 'aeiou'
    c = float(sum([1 for ch in s if ch in punc]))
    w = ((c/float(len(s)))) * 100
    w = int(w)
    #return int(w/float(bins))
    return "VOWEL_RATIO_[{}]".format( int(w/float(bins)) )



def word_soundex(c,idxs):
    s = c.get_attrib_span("words")
    tokens = s.split()
    for i in range(0,len(tokens)):
        seq = map(soundex,tokens[0:i+1])
        yield "SOUNDEX_SEQ_[{}]".format(" ".join(seq))



def affex_norm(affex):
    affex = affex.lower() 
    if affex.isdigit():
        affex = "D"
    elif affex in string.punctuation:
        affex = "P"
    return affex

def affexes(c,idxs):
    s = c.get_attrib_span("words")
    tokens = s.split()
    
    seq = morphology.inserted(tokens[0])
    t = seq.split("-")
    yield "PREFIX_FW_[{}]".format(affex_norm(t[0]))
    if len(t) > 1:
        yield "SUFFIX_FW_[{}]".format(affex_norm(t[-1]))
          
    if len(tokens) > 1:
        seq = morphology.inserted(tokens[-1])
        t = seq.split("-")
        yield "PREFIX_LW_[{}]".format(affex_norm(t[0]))
        if len(t) > 1:
            yield "SUFFIX_LW_[{}]".format(affex_norm(t[-1]))
        
        
   
def affexes2(c,idxs):
    s = c.get_attrib_span("words")
    ftr = morphology.inserted(s)
    t = ftr.split("-")
    yield "PREFIX_[{}]".format(t[0].lower())
    if len(t) > 1:
        yield "SUFFIX_[{}]".format(t[-1].lower())


def word_shape(c,idxs):
    s = c.get_attrib_span("words")
    
    if len(s) >= 100:
        yield 'LONG'
    length = len(s)
    shape = []
    last = ""
    shape_char = ""
    seq = 0
    for c in s:
        if c.isalpha():
            if c.isupper():
                shape_char = "X"
            else:
                shape_char = "x"
        elif c.isdigit():
            shape_char = "d"
        else:
            shape_char = c
        if shape_char == last:
            seq += 1
        else:
            seq = 0
            last = shape_char
        if seq < 4:
            shape.append(shape_char)
    
    yield "[{}]".format(''.join(shape))


def generate_mention_feats(get_feats, prefix, candidates):
    for i,c in enumerate(candidates):
        for ftr in get_feats(c):
            yield i, prefix + ftr


class Featurizer(object):
    """
    A Featurizer applies a set of **feature generators** to each Candidate,
    based on (i) the arity of the candidate, and (ii) the _associated Contexts_.

    The transform() function takes in N candidates, and returns an N x F sparse matrix,
    where F is the dimension of the feature space.
    """
    def __init__(self, arity=1):
        self.arity          = arity
        self.feat_index     = None
        self.feat_inv_index = None

    def _generate_context_feats(self, get_feats, prefix, candidates):
        """
        Given a function that given a candidate, generates features, _using a specific context_,
        and a unique prefix string for this context, return a generator over features (as strings).
        """
        for i,c in enumerate(candidates):
            for f in get_feats(c):
                yield i, prefix + f

    # TODO: Take this out...
    def _preprocess_candidates(self, candidates):
        return candidates

    def _match_contexts(self, candidates):
        """Given the candidates, and using _generate_context_feats, return a list of generators."""
        raise NotImplementedError()

    def transform(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F                  = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))
        for i,f in itertools.chain(*feature_generators):
            if self.feat_index.has_key(f):
                F[i,self.feat_index[f]] = 1
        return F

    def fit_transform(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))

        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        for i,f in itertools.chain(*feature_generators):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index     = {}
        self.feat_inv_index = {}
        F                   = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feat_index[f] = j
            self.feat_inv_index[j] = f
            for i in f_index[f]:
                F[i,j] = 1
        return F

    def get_features_by_candidate(self, candidate):
        feature_generators = self._match_contexts(self._preprocess_candidates([candidate]))
        feats = []
        for i,f in itertools.chain(*feature_generators):
            feats.append(f)
        return feats

    def top_features(self, w, n_max=100):
        """Return a DataFrame of highest (abs)-weighted features"""
        idxs = np.argsort(np.abs(w))[::-1][:n_max]
        d = {'j': idxs, 'w': [w[i] for i in idxs]}
        return DataFrame(data=d, index=[self.feat_inv_index[i] for i in idxs])

class NgramFeaturizer(Featurizer):
    """Feature for relations (of arity >= 1) defined over Ngram objects."""
    def _preprocess_candidates(self, candidates):
        for c in candidates:
            if not isinstance(c.sentence, dict):
                c.sentence = get_as_dict(c.sentence)
            if c.sentence['xmltree'] is None:
                c.sentence['xmltree'] = corenlp_to_xmltree(c.sentence)
        return candidates

    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, range(c.word_start, c.word_end+1)), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            get_feats = compile_entity_feature_generator()
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_feats(c.sentence['xmltree'].root, range(c.word_start, c.word_end+1)), 'TDL_', candidates))
            
            # word shape features
            feature_generators.append( generate_mention_feats( \
                lambda c: word_shape(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # soundex
            feature_generators.append( generate_mention_feats( \
                lambda c: word_soundex(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # affexes
            feature_generators.append( generate_mention_feats( \
                lambda c: affexes(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # letter ratio
            #feature_generators.append( generate_mention_feats( \
            #    lambda c: letter_ratio(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # vowel ratio
            #feature_generators.append( generate_mention_feats( \
            #    lambda c: vowel_ratio(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            
        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")
        return feature_generators


class LegacyCandidateFeaturizer(Featurizer):
    """Temporary class to handle v0.2 Candidate objects."""
    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, c.idxs), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            if candidates[0].root is not None:
                get_feats = compile_entity_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.idxs), 'TDLIB_', candidates))

        if self.arity == 2:

            # Add TreeDLib relation features
            if candidates[0].root is not None:
                get_feats = compile_relation_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.e1_idxs, c.e2_idxs), 'TDLIB_', candidates))
        return feature_generators

