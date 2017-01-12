import re
import numpy as np
import codecs
from candidates import *
from gensim.models.word2vec import Word2Vec

class DictionarySeqFeaturizer(object):
    '''Should do this at the sentence level for CRF so that we
    capture multiple word spans'''
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def _sentence_match(self, sent, dictionary, max_ngrams=4):
        # TODO setup to match raw span vs tokens
        words = zip(*sent)[0]
        tags = [0] * len(words)
        spans = {}  # longest span match
        for i in range(len(words)):
            if tags[i] == 1:
                continue
            span_len = i + 1
            for j in range(i + 1, min(len(words), i + 1 + max_ngrams)):
                term = " ".join(words[i:j]).lower()
                if term in dictionary:
                    tags[i:j] = [1] * (j - i)
                    spans[i] = term
                    span_len = j

            if tags[i] == 1:
                term = spans[i]
                for j in range(i, span_len):
                    spans[j] = term

        return tags, spans


    def get_ftrs(self, c):
        #dict_tags, dict_spans = dictionary_match(tokens, dict_diseases)
        # ftrs += ['word.dictionary=%s' % (dict_tags[i] == 1)]
        # if i in dict_spans:
        #     ftrs += ['word.span=%s' % dict_spans[i].replace(" ", "_")]
        pass


class DictionaryFeaturizer(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def get_ftrs(self, c):
        term = c.get_attrib_span("words")
        if term.lower() in self.dictionary or term in self.dictionary:
            yield 'word.dictionary=True'













# ---------------------
# Sequence Features
#
class CharNgramFeaturizer(object):
    def get_ftrs(self, word):
        for i in range(len(word) - 1):
            yield "word.char_ngram=" + word[i:i + 2]
        for i in range(len(word) - 2):
            yield "word.char_ngram=" + word[i:i + 3]


class WordShapeFeaturizer(object):
    def get_ftrs(self, s):
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


class EmbeddingFeaturizer(object):
    '''Load and binarize embeddings'''
    def __init__(self, model, fmt="gensim"):

        self._load_model(fmt, model)
        self._binarize()


    def _load_model(self, fmt, emb_model):

        if fmt == "gensim":
            model = Word2Vec.load(emb_model) if type(emb_model) is str else emb_model
            model.init_sims()

            self.index2word = {model.vocab[word].index: word for word in model.vocab}
            self.word2index = {word: model.vocab[word].index for word in model.vocab}
            self.emb = model.syn0norm

        elif fmt == "text":
            emb = []
            self.index2word = {}
            m = 0
            with open(emb_model,"rU") as fp:
                i,malformed = 0,0
                for line in fp:
                    row = line.strip().split()
                    term,vec = row[0],row[1:]
                    if i == 0:
                        m = len(vec)
                    if len(vec) != m:
                        malformed += 1
                        print line[0:200]
                        print term
                        print len(vec)
                        continue
                    self.index2word[i] = term
                    emb += [[float(v) for v in vec]]
                    i += 1
            if malformed > 0:
                print>> sys.stderr, "WARNING {} malformed rows".format(malformed)

            self.word2index = {word:idx for idx,word in self.index2word.items()}
            self.emb = np.array(emb)
            print self.emb.shape

        elif fmt == "word2vec":
            print "Warning word2vec binary not implemented!"


    def _binarize(self):
        '''Simple transform of continuous dense embedding to binary
        For each dimension D X Vocabulary
         1) Split into pos/neg parts
         2) Compute mean of pos/neg
         3) v+ >= mean+   =  1
            v- <= mean-   = -1
            otherise      =  0
        creating 2 x D additional features
        '''
        emb_T = self.emb.T
        emb = np.zeros(emb_T.shape, dtype=np.int8)
        print emb.shape

        for i in range(emb_T.shape[0]):
            pos_mean = np.mean(emb_T[i][emb_T[i] > 0])
            neg_mean = np.mean(emb_T[i][emb_T[i] < 0])

            emb[i][emb_T[i] >= pos_mean] = 1
            emb[i][emb_T[i] <= neg_mean] = -1

        self.emb = emb.T


    def _best_match(self,c):
        '''Attempt some common transforms to find the closest
        matching term with a learned representation'''
        term = c.get_attrib_span("words")

        if term in self.word2index:
            return self.word2index[term]
        if term.lower() in self.word2index:
            return self.word2index[term.lower()]

        term = re.sub("[.=/!?]+","",term)
        if term in self.word2index:
            return self.word2index[term]
        if term.lower() in self.word2index:
            return self.word2index[term.lower()]
        return -1

    def get_ftrs(self, c):

        i = self._best_match(c)
        if i != -1:
            row = self.emb[i]
            for i in range(row.shape[0]):
                yield "emb_{}={}".format(i, row[i])


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
    def __init__(self, documents, featurizer):
        self.accept_rgx = '[0-9A-Z-]{2,8}[s]*'
        self.reject_rgx = '([0-9]+/[0-9]+|[0-9]+[-][0-7]+)'
        
        self.docs = documents
        self.short_form_index = self.get_short_form_index(self.docs)
        self.ftr_index = {doc_id:{sf:[] for sf in self.short_form_index[doc_id]} for doc_id in self.short_form_index } #sf_index[doc.doc_id][short_form]

        self.featurizer = featurizer
        
        # compute features for each short form
        for doc_id in self.short_form_index:
            for sf in self.short_form_index[doc_id]:
                ftrs = []
                for lf in self.short_form_index[doc_id][sf]:
                    ftrs += self.get_short_form_ftrs(lf)
                self.ftr_index[doc_id][sf] = list(set(ftrs))

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

    def get_short_form_ftrs(self,c):
        ftrs = list(self.featurizer.get_ftrs(c))
        ftrs = [f for f in ftrs if f not in ["BOS","EOS"] and not re.search("^[+-]*1:",f)]
        ftrs += ["word.span=%s" % c.get_attrib_span("words").lower().replace(" ", "_")]
        return ftrs

    def get_ftrs(self,c):
        word = c.get_attrib_span("words")
        w_ftrs = []
        if c.doc_id in self.ftr_index and word in self.ftr_index[c.doc_id]:
            w_ftrs += list(self.ftr_index[c.doc_id][word])

        for i,ftr in enumerate(w_ftrs):
            yield ftr


