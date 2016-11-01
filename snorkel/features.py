import os, sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
import itertools
from pandas import DataFrame
import re
import time
# Feature modules
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from utils import get_as_dict
from entity_features import *

import math
import numpy as np
from multiprocessing import Process, Queue

import string
import fuzzy
import pyphen
import re
import codecs
from candidates import *

#pyphen.language_fallback('nl_NL_variant1')
morphology = pyphen.Pyphen(lang="en_Latn_US")
#morphology = pyphen.Pyphen(lang="en_US")
morphology = pyphen.Pyphen(lang="nl_NL_variant1")

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


def word_seq_affixes(c,idxs):
    # morphology assumptions different for acronyms/abbreviations
    s = c.get_attrib_span("words")
    #if re.search("^[0-9A-Z-]{2,8}[a-z]{0,1}$",s):
    #    return

    tokens = s.split()
    for t in tokens:
        m = morphology.inserted(t).split("-")
        if len(m) == 1:
            yield u"MORPHEME_FREE_[{}]".format(affex_norm(m[0]))
        else:
            yield u"MORPHEME_PREFIX_[{}]".format(affex_norm(m[0]))
            yield u"MORPHEME_SUFFIX_[{}]".format(affex_norm(m[-1]))
            if len(m) > 2:
                root = "".join(m[1:-1])
                yield u"MORPHEME_ROOT_[{}]".format(affex_norm(root))
                

def word_shape_seq(c,idxs):
    words = c.get_attrib_span("words")
    yield u"[{}]".format(word_shape(words)) 
    
    tokens = words.split()
    if len(tokens) > 1:
        for w in tokens:
            yield u"SEQ_[{}]".format(word_shape(w))
    

def word_shape(s):
    '''From SpaCY'''
    if len(s) >= 100:
        return 'LONG'
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
    
    return ''.join(shape)
   

def left_window(m, window=3, match_attrib="lemmas"):
    idx = max(0,min(m.idxs) - window)
    span = range(idx,min(m.idxs))
    return [m.get_attrib(match_attrib)[i] for i in span]


def right_window(m, window=3, match_attrib="lemmas"):
    idx = min(len(m.get_attrib()), max(m.idxs) + window + 1)
    span = range(max(m.idxs) + 1,idx)
    return [m.get_attrib(match_attrib)[i] for i in span]

rgx_is_digit = re.compile("([0-9]+[,.]*)+")

def word_seq(c,idxs):
    '''Linear chain within mention'''
    words = c.get_attrib_tokens("lemmas")
    lw = left_window(c,window=1)
    rw = right_window(c,window=1)
    lw = u"_" if not lw else lw[0]
    rw = u"_" if not rw else rw[0]
    lw = u"NUMBER" if rgx_is_digit.search(lw) else lw
    rw = u"NUMBER" if rgx_is_digit.search(rw) else rw
    
    for i in range(len(words)):
        left = rw if i == 0 else words[i-1]
        right = lw if i == len(words) - 1 else words[i+1]
        yield u"LEMMA_L_[{}]".format(left)
        yield u"LEMMA_R_[{}]".format(right)
   
   
def morpheme_seq(c,idxs,ngr=2):
    # abbrevations don't have meaningful morphological units
    #s = c.get_attrib_span("words")
    #if re.search("^[0-9A-Z-]{2,8}[a-z]{0,1}$",s):
    #    return
    
    s = c.get_attrib_span("lemmas")
    tokens = s.split()
    
    tmpl = u"MORPHEME_SEQ_[{}]"
    for i in range(0,len(tokens)):  
        seq = morphology.inserted(tokens[i]).split("-")
        seq = map(lambda x:re.sub("\d",u"D",x.lower()),seq)
        seq = map(lambda x:re.sub("[.()\]\['-]",u"P",x.lower()),seq)
        if len(seq) > 1:
            for j in range(0,len(seq)-ngr+1):
                v = u"" if j == 0 else u"-"
                v += u"".join(seq[j:j+2])
                v += u"" if j+2 == len(seq) else u"-"
                yield tmpl.format(v)


import codecs

def binary_mention_features(c,idxs):
    s = c.get_attrib_span("words")
    
    if s.isupper():
        yield u"ALL_UPPERCASE"

    if re.search("[0-9]+",s):
        yield u"CONTAINS_DIGITS"

    if re.search("[.:;'/()\[\]-]+",s):
        yield u"CONTAINS_PUNCTUATION"
        
    lw = left_window(c,window=1)
    rw = right_window(c,window=1)
    lw = u"_" if not lw else lw[0]
    rw = u"_" if not rw else rw[0]
    
    if lw in ["-lrb-","(",'-LRB-'] and rw in ["-rrb-",")","-RRB-"]:
        yield u"PARANTHETICAL"
    
    ##lw = left_window(c,window=8)
    #lw = " ".join(lw).lower()
    #lw = lw.replace("-lrb-","(").replace("-rrb-",")")
    #if re.search("[(]\s*.+?\s*[)]",lw):
    #    yield u"RIGHT_OF_PARANTHETICAL"
        




def generate_mention_feats(get_feats, prefix, candidates):
    for i,c in enumerate(candidates):
        for ftr in get_feats(c):
            yield i, prefix + ftr

            
class FeaturizerMP(object):
    
    def __init__(self, num_procs=1):
        self.num_procs      = num_procs
        self.feat_index     = None
        self.feat_inv_index = None
    
    @staticmethod
    def featurizer_worker(pid,idxs,candidates,queue): 
        print "\tFeaturizer process_id={} {} items".format(pid, len(idxs))
        block = [candidates[i] for i in idxs]
        feature_generators = FeaturizerMP.apply(block)
        ftr_index = defaultdict(list)
        for i,ftr in itertools.chain(*feature_generators):
            ftr_index[ftr].append(idxs[i])
        #queue.put(ftr_index)
        outdict = {pid:ftr_index}
        queue.put(outdict)
    
    @staticmethod
    def generate_feats(get_feats, prefix, candidates):
        for i,c in enumerate(candidates):
            for f in get_feats(c):
                yield i, prefix + f
   
    @staticmethod
    def preprocess(candidates):
        for c in candidates:
            if not isinstance(c.sentence, dict):
                c.sentence = get_as_dict(c.sentence)
            if c.sentence['xmltree'] is None:
                c.sentence['xmltree'] = corenlp_to_xmltree(c.sentence)
        return candidates
    
    @staticmethod
    def get_features_by_candidate(candidate):
        feature_generators = FeaturizerMP.apply(FeaturizerMP.preprocess([candidate]))
        feats = []
        for i,f in itertools.chain(*feature_generators):
            feats.append(f)
        return feats

    @staticmethod
    def apply(candidates):
        
        feature_generators = []
        
        # Add DDLIB entity features
        feature_generators.append(FeaturizerMP.generate_feats( \
            lambda c : get_ddlib_feats(c, range(c.word_start, c.word_end+1)), 'DDLIB_', candidates))

        # Add TreeDLib entity features
        get_feats = compile_entity_feature_generator()
        feature_generators.append(FeaturizerMP.generate_feats( \
            lambda c : get_feats(c.sentence['xmltree'].root, range(c.word_start, c.word_end+1)), 'TDL_', candidates))
        
        # word shape features
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: word_shape_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
        
        # soundex
        #feature_generators.append( generate_mention_feats( \
        #    lambda c: word_soundex(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
         
        # morphemes
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: morpheme_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
        
        # affixes
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: word_seq_affixes(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
        
        # mention word linear chain
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: word_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
       
        # binary mention features
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: binary_mention_features(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
             
        return feature_generators

    def top_features(self, w, n_max=100):
        """Return a DataFrame of highest (abs)-weighted features"""
        idxs = np.argsort(np.abs(w))[::-1][:n_max]
        d = {'j': idxs, 'w': [w[i] for i in idxs]}
        return DataFrame(data=d, index=[self.feat_inv_index[i] for i in idxs])
    
    def fit(self,candidates):
        
        self.feat_index = {}
        self.feat_inv_index = {}
        candidates = FeaturizerMP.preprocess(candidates)
        
        if self.num_procs > 1:    
            
            out_queue = Queue()
            chunksize = int(math.ceil(len(candidates) / float(self.num_procs)))
            procs = []

            nums = range(0,len(candidates))
            for i in range(self.num_procs):
                p = Process(
                            target=FeaturizerMP.featurizer_worker,
                            args=(i, nums[chunksize * i:chunksize * (i + 1)],
                                  candidates,
                                  out_queue))
                procs.append(p)
                p.start()

            resultdict = {}
            for i in range(self.num_procs):
                resultdict.update(out_queue.get())
            
            # merge feature    
            #f_index = defaultdict(list)
            #for i in range(self.num_procs):
            #    block = out_queue.get()
            #    for ftr in block:
            #        f_index[ftr].extend(block[ftr])
            
            for p in procs:
                p.join()
        
            # merge feature    
            f_index = defaultdict(list)
            for i in resultdict: 
                for ftr in resultdict[i]:
                    f_index[ftr].extend(resultdict[i][ftr])
        
        else:
            feature_generators = FeaturizerMP.apply(candidates)
            f_index = defaultdict(list)
            for i,f in itertools.chain(*feature_generators):
                f_index[f].append(i)
        
        for j,f in enumerate(sorted(f_index.keys())):
            self.feat_index[f] = j
            self.feat_inv_index[j] = f
        
        self.f_index = f_index
        
    def fit_transform(self, candidates):
        self.fit(candidates)
        return self.transform(candidates)
    
    def transform(self,candidates):
        if not self.f_index:
            raise Exception('model is not fit')
        
        F = sparse.lil_matrix((len(candidates), len(self.f_index.keys())))
        for f in sorted(self.f_index.keys()):
            j = self.feat_index[f]
            for i in self.f_index[f]:
                F[i,j] = 1
        
        return F
        
        


class Featurizer(object):
    """
    A Featurizer applies a set of **feature generators** to each Candidate,
    based on (i) the arity of the candidate, and (ii) the _associated Contexts_.

    The transform() function takes in N candidates, and returns an N x F sparse matrix,
    where F is the dimension of the feature space.
    """
    def __init__(self, arity=1, cluster_defs=None):
        self.arity          = arity
        self.feat_index     = None
        self.feat_inv_index = None
        self.cluster_defs = cluster_defs
        self.kmeans_ftrs = None
        
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

    #Featurizer._match_contexts(self._preprocess_candidates(candidates))


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
        for j,f in enumerate(sorted(f_index.keys())):
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
    def __init__(self, use_acronym_ftrs=True, use_kmeans_ftrs=False,
                 arity=1, cluster_defs=None):
        super(NgramFeaturizer, self).__init__(arity,cluster_defs=None)
        self.use_acronym_ftrs = use_acronym_ftrs
        self.use_kmeans_ftrs = use_kmeans_ftrs
        #print "Using Acronym Features", self.use_acronym_ftrs
        #print "Using kmeans Features", self.use_kmeans_ftrs
       
    def _preprocess_candidates(self, candidates):
        for c in candidates:
            if not isinstance(c.sentence, dict):
                c.sentence = get_as_dict(c.sentence)
            if c.sentence['xmltree'] is None:
                c.sentence['xmltree'] = corenlp_to_xmltree(c.sentence)
        return candidates

    def _match_contexts(self, candidates):
        feature_generators = []
        
        if self.use_acronym_ftrs:
            acronym_ftrs = AcronymFeaturizer(candidates)
        
        if self.kmeans_ftrs == None and self.cluster_defs:
            self.kmeans_ftrs = KMeansFeaturizer(self.cluster_defs)
            
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
                lambda c: word_shape_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # ***
            # soundex
            #feature_generators.append( generate_mention_feats( \
            #    lambda c: word_soundex(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
             
            # morphemes
            feature_generators.append( generate_mention_feats( \
                lambda c: morpheme_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # affixes
            feature_generators.append( generate_mention_feats( \
                lambda c: word_seq_affixes(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # mention word linear chain
            feature_generators.append( generate_mention_feats( \
                lambda c: word_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )

            # mention word linear chain
            feature_generators.append( generate_mention_feats( \
                lambda c: binary_mention_features(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )            
            
            # acronym features
            if self.use_acronym_ftrs:
                feature_generators.append( generate_mention_feats( acronym_ftrs.get_ftrs, "", candidates) )            
            
            # kmeans features
            if self.use_kmeans_ftrs:
                feature_generators.append( generate_mention_feats( self.kmeans_ftrs.get_ftrs, "", candidates) )            
            
            
 
 
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


class KMeansFeaturizer(object):
    def __init__(self,cluster_defs):
        self.word2cluster = self._load_cluster_defs(cluster_defs)
    
    def _load_cluster_defs(self, filename):
        d = {}
        for line in codecs.open(filename, 'r', 'utf-8'):
            cid, w = line.strip().split('\t')
            words = w.split('|')
            for w in words:
                d[w] = cid
        return d

    def get_ftrs(self,c):
        tokens = c.get_attrib_tokens("lemmas")
        for t in tokens:
            if t.lower() in self.word2cluster:
                ftr = 'WORD_CLUSTER_' + str(self.word2cluster[t.lower()])
                yield ftr
        
class AcronymFeaturizer(object):
    '''Requires document-level knowledge
    '''
    def __init__(self,candidates):
        self.accept_rgx = '[0-9A-Z-]{2,8}[s]*'
        self.reject_rgx = '([0-9]+/[0-9]+|[0-9]+[-][0-7]+)'
        
        self.docs = [c.metadata["doc"] for c in candidates if "doc" in c.metadata]
        self.short_form_index = self.get_short_form_index(self.docs)
        self.ftr_index = {doc_id:{sf:[] for sf in self.short_form_index[doc_id]} for doc_id in self.short_form_index } #sf_index[doc.doc_id][short_form]
        #self.global_ftr_index = {doc_id:{sf:[] for sf in self.short_form_index[doc_id]} for doc_id in self.short_form_index } #sf_index[doc.doc_id][short_form]
        
        self.featurizer = NgramFeaturizer(use_acronym_ftrs=False)
        
        # compute features for each short form
        for doc_id in self.short_form_index:
            for sf in self.short_form_index[doc_id]:
                ftrs = []
                for lf in self.short_form_index[doc_id][sf]:
                    ftrs += self.get_short_form_ftrs(lf)
                self.ftr_index[doc_id][sf] = list(set(ftrs))
                #print sf, len(self.ftr_index[doc_id][sf]), self.ftr_index[doc_id][sf][0:10]
                #print
                #if sf not in self.global_ftr_index:
                #    self.global_ftr_index[sf] = []
                #self.global_ftr_index[sf] += self.ftr_index[doc_id][sf]
        
        #for sf in self.global_ftr_index:
        #    print sf, len(self.global_ftr_index[sf]), self.global_ftr_index[sf]

        
        

    def is_short_form(self, s, min_length=2):
        '''
        Rule-based function for determining if a token is likely
        an abbreviation, acronym or other "short form" mention
        TODO: extend to anything inside parantheses? Too noisy?
        '''
        keep = re.search(self.accept_rgx,s) != None
        keep &= re.search(self.reject_rgx,s) == None
        keep &= not s.strip("-").isdigit()
        keep &= "," not in s
        keep &= len(s) < 15
        
        # reject?
        reject = (len(s) > 3 and not keep) # regex reject strings of len > 3
        reject |= (len(s) <= 3 and re.search("[/,+0-9-]",s) != None) # contains junk chars
        reject |= (len(s) < min_length) # too short
        reject |= (len(s) <= min_length and s.islower()) # too short + lowercase single letters
        
        return False if reject else True

    def get_parenthetical_short_forms(self, sentence):
        '''
        Generator that returns indices of all words 
        directly wrapped by paranthesis or brackets
        '''
        for i,w in enumerate(sentence.words):
            if i > 0 and i < len(sentence.words) - 1:
                window = sentence.words[i-1:i+2]
                if (window[0] == "(" and window[-1] == ")"):
                    if self.is_short_form(window[1]):
                        yield i
        
    def extract_long_form(self, i, sentence, max_dup_chars=2):
        '''
        Search the left window for a candidate long-form sequence.
        Use the hueristic of "match first character" to guess long form
        '''
        short_form = sentence.words[i]
        left_window = [w for w in sentence.words[0:i]]
        
        # strip brackets/parantheses
        while left_window and left_window[-1] in ["(","[",":"]:
            left_window.pop()
           
        if len(left_window) == 0:
            return None
        
        # match longest seq to the left of our short form
        # that matches on starting character
        long_form = []
        char = short_form[0].lower()
        letters = [t[0].lower() for t in short_form]
        letters = [t for t in letters if t == char]
        letters = letters[0:min(len(letters),max_dup_chars)]
        
        matched = False
        for t in left_window[::-1]:
            if t[0] in "()[]-+,":
                break
            if len(letters) == 1 and t[0].lower() == letters[0]:
                long_form += [t]
                matched = True
                break
            elif len(letters) > 1 and t[0].lower() == letters[0]:
                long_form += [t]
                matched = True
                letters.pop(0)
            else:
                long_form += [t]
        
        # we didn't find the first letter of our short form, so 
        # backoff and choose the longest contiguous noun phrase
        if (len(left_window) == len(long_form) and \
           letters[0] != t[0].lower() and len(long_form[::-1]) > 1) or not matched:
            
            tags = zip(sentence.words[0:i-1],sentence.poses[0:i-1])[::-1]
            noun_phrase = []
            while tags:
                t = tags.pop(0)
                if re.search("^(NN[PS]*|JJ)$",t[1]):
                    noun_phrase.append(t)
                else:
                    break
                    
            if noun_phrase:
                long_form = zip(*noun_phrase)[0]
       
        # create candidate
        n = len(long_form[::-1])
        offsets = sentence.char_offsets[0:i-1][-n:]
        char_start = min(offsets)
        words = sentence.words[0:i-1][-n:]
        pos_tags = sentence.poses[0:i-1][-n:]
        offsets = map(lambda x:len(x[0])+x[1], zip(words,offsets))
        char_end = max(offsets)
        return Ngram(char_start, char_end-1, sentence, {"short_form":short_form}) 


    def get_short_form_index(self, documents):
        '''
        Build a short_form->long_form mapping for each document. Any 
        short form (abbreviation, acronym, etc) that appears in parenthetical
        form is considered a "definiton" and added to the index. These candidates
        are then used to augment the features of future mentions with the same
        surface form.
        '''
        sf_index = {}
        for doc in documents:
            for sent in doc.sentences:
                for i in self.get_parenthetical_short_forms(sent):
                    short_form = sent.words[i]
                    long_form_cand = self.extract_long_form(i,sent)
                    
                    if not long_form_cand:
                        continue
                    if doc.doc_id not in sf_index:
                        sf_index[doc.doc_id] = {}
                    if short_form not in sf_index[doc.doc_id]:
                        sf_index[doc.doc_id][short_form] = []
                    sf_index[doc.doc_id][short_form] += [long_form_cand]
                    
        return sf_index

    def get_ftrs(self,c):
        word = c.get_attrib_span("words")
        w_ftrs = []
        if c.doc_id in self.ftr_index and word in self.ftr_index[c.doc_id]:
            w_ftrs += list(self.ftr_index[c.doc_id][word])
           
        for i,ftr in enumerate(w_ftrs):
            yield ftr
        
            
    def get_short_form_ftrs(self,c):
        '''Hack to filter out some features'''
        ftrs = self.featurizer.get_features_by_candidate(c)
        f_ftrs = []
        #rgx = "^(DDLIB_(WORD|LEMMA|POS|DEP)_SEQ_|TDL_|WS_)" 
        rgx = "^(DDLIB_(WORD|LEMMA|POS|DEP)_SEQ_|WS_)" 
        for f in ftrs:
            if not re.search(rgx,f):
                continue
            if "WS_LEMMA_" in f:
                continue
            f_ftrs += [f]
            
        tokens = c.get_attrib_tokens("lemmas")
        for t in tokens:
            f_ftrs += ["TDL_LEMMA:MENTION[{}]".format(t)]
        f_ftrs += ["WS_SHORT_FORM_DEFINED"]
        return list(set(f_ftrs))
