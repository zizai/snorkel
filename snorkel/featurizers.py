import re
import codecs
from candidates import *


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
                #ftr = str(self.word2cluster[t.lower()])
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


    def get_ftrs(self,c):
        word = c.get_attrib_span("words")
        w_ftrs = []
        if c.doc_id in self.ftr_index and word in self.ftr_index[c.doc_id]:
            w_ftrs += list(self.ftr_index[c.doc_id][word])
           
        for i,ftr in enumerate(w_ftrs):
            yield ftr


    def get_short_form_ftrs(self,c):
        ftrs = list(self.featurizer.get_ftrs(c))

        ftrs = [f for f in ftrs if f not in ["BOS","EOS"] and not re.search("^[+-]*1:",f)]
        #ftrs += ["word.lower=%s" % c.get_attrib_span("words").lower().replace(" ","_")]
        ftrs += ["word.span=%s" % c.get_attrib_span("words").lower().replace(" ", "_")]

        # just add word tokens
        #for t in c.get_attrib_span("words").lower().split():
        #   ftrs += ["word.lower=%s" % t]

        return ftrs
