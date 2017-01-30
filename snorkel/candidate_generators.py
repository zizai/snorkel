import re
import os
import bz2
import sys
import math
import codecs
import cPickle
import itertools
import numpy as np
from nltk.util import ngrams
from collections import defaultdict

from candidates import Candidates, Candidate, Ngrams, Ngram
from matchers import DictionaryMatch, Union, Concat
from matchers import RegexMatchSpan, RegexMatchEach, KgramMatcher, SlotFillMatch

from ddbiolib.datasets import ncbi_disease
from ddbiolib.ontologies.umls import UmlsNoiseAwareDict
from ddbiolib.ontologies.ctd import load_ctd_dictionary
from ddbiolib.ontologies.specialist import SpecialistLexicon
from ddbiolib.ontologies.bioportal import load_bioportal_dictionary
from ddbiolib.ontologies.umls.dictionary import UmlsDictionary
from ddbiolib.ontologies.ctd import load_ctd_dictionary
from ddbiolib.ontologies.bioportal import load_bioportal_dictionary
from ddbiolib.ontologies.specialist import SpecialistLexicon

# notebook helper functions
from experiments.utils import *


def get_umls_stopwords(n_max=1, keep_words={}):
    stop_entity_types = ["Quantitative Concept", "Temporal Concept", "Animal", "Food",
                         "Spatial Concept", "Functional Concept"]
    stop_entities = UmlsNoiseAwareDict(positive=stop_entity_types,
                                       name="terms", ignore_case=True).dictionary()
    d = {t: 1 for t in stop_entities if t not in keep_words and len(t.split()) <= n_max}
    return d


class Tfidf(object):

    def __init__(self,corpus,idf_set):
        self.corpus = corpus
        self.num_docs = len(idf_set)
        self.doc_tf_cache = {}
        self.df = self._compute_doc_freq({doc_id:self.corpus[doc_id] for doc_id in idf_set})

    def _compute_doc_freq(self,documents):
        # compute term df using the *training set* for 1-2grams
        df = {}
        for doc_id in documents:
            doc_tf = self.term_freq(documents[doc_id])
            for t in doc_tf:
                df[t] = df.get(t, 0.0) + 1
        return df

    def term_freq(self, doc):
        doc_tf = {}
        for sentence in doc.sentences:
            tokens = map(lambda x: x.lower(), sentence.lemmas)
            for t in tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + 1.0
            for t in ngrams(tokens, 2, pad_left=False, pad_right=False):
                t = " ".join(t)
                doc_tf[t] = doc_tf.get(t, 0.0) + 1.0
        return doc_tf

    def get_tf_idf(self, doc_id, mention):

        doc = self.corpus[doc_id]

        doc_tf = self.term_freq(doc) if doc_id not in self.doc_tf_cache else self.doc_tf_cache[doc_id]
        self.doc_tf_cache[doc_id] = doc_tf

        idf = math.log(self.num_docs / float(self.df[mention])) if mention in self.df else math.log(self.num_docs / 1.0)

        # augmented tf
        tf = doc_tf[mention] / max(doc_tf.values()) if mention in doc_tf else 1.0 / max(doc_tf.values())
        tf = 0.5 + (0.5 * tf)  # weighted
        return tf * idf


class CandidateGenerator(object):
    def __init__(self,kgrams, split_chars):
        self.kgrams = kgrams
        self.split_chars = split_chars


# ------------------------------------------------
#
#  K-Grams
#
# ------------------------------------------------
class KGramsGenerator(CandidateGenerator):

    def __init__(self, kgrams, split_chars=['-', '/', '\+']):

        super(KGramsGenerator, self).__init__(kgrams, split_chars)
        self.cand_space = Ngrams(n_max=self.kgrams, split_tokens=self.split_chars)
        self.matcher = KgramMatcher(k=self.kgrams)

    def get_candidates(self, sentences, min_tfidf=1.0,
                       fuzzy_match=False, nprocs=1):
        '''
        Extract candidates
        :param sentences:  sentences to parse
        :param nprocs:  mulit-processing
        :return:
        '''
        cs = Candidates(self.cand_space, self.matcher, sentences, parallelism=nprocs)
        candidates = cs.get_candidates()
        return candidates


# ------------------------------------------------
#
#  Noun Phrase (Singularity)
#
# ------------------------------------------------
class NounPhraseGenerator(CandidateGenerator):
    '''
    Restrict candidates to heuristically identified noun phrases
    '''
    def __init__(self, kgrams, split_chars=['-', '/', '\+'],
                 skipwords = ["'" ,"'s" ,"de" ,"von"], tfidf=None):
        super(NounPhraseGenerator, self).__init__(kgrams,split_chars)
        self.cand_space = Ngrams(n_max=self.kgrams, split_tokens=self.split_chars)
        self.matcher = KgramMatcher(k=self.kgrams)
        self.skipwords = set(skipwords)
        self.tfidf = tfidf

        #self.rgx_np = re.compile("^((JJ[S]*|NN[PS]*|FW)\s*){1,}$")
        self.np_tags = set(['NN', 'NNP', 'NNS', 'NNPS', 'FW', 'JJ', 'JJS'])

    def get_candidates(self, sentences, min_tfidf=1.0,
                       fuzzy_match=False, nprocs=1):
        '''
        Extract candidates
        :param sentences:  sentences to parse
        :param nprocs:  mulit-processing
        :param fuzzy_match: allow non-contiguous noun phrases
        :return:
        '''
        cs = Candidates(self.cand_space, self.matcher, sentences, parallelism=nprocs)
        candidates = cs.get_candidates()

        # filter to noun phrases
        if not fuzzy_match:
            candidates = [c for c in candidates if self._is_noun_phrase(c)]
            if self.tfidf != None:
                candidates = self._tfidf_filter(candidates, self.tfidf, threshold=min_tfidf)
        else:
            candidates = [c for c in candidates if self._is_noun_phrase(c) or self._is_bookended_noun_phrase(c)]
        return candidates

    def _tfidf_filter(self,candidates, tfidf, threshold=1.0):

        f_candidates = []
        for i, c in enumerate(candidates):
            mention = " ".join(c.get_attrib_tokens("words"))
            w = tfidf.get_tf_idf(c.doc_id, mention)
            if w > threshold:
                f_candidates += [c]

        return f_candidates

    def _is_noun_phrase(self,c):
        '''
        Test if candidate is a noun phrase
        :param c: candidate
        :return: boolean
        '''
        pos_tags = c.get_attrib_tokens("poses")
        words = c.get_attrib_tokens("words")
        v = [1 for t,w in zip(pos_tags ,words) if t in self.np_tags or (w in self.skipwords and len(pos_tags) > 2)]
        return len(v) == len(pos_tags)

    def _is_bookended_noun_phrase(self,c):
        '''
        Fuzzy noun phrase matching (allow some tags between NN[PS]* tags)
        :param c: candidate
        :return: boolean
        '''
        pos_tags = c.get_attrib_tokens("poses")
        head = pos_tags[0]
        tail = pos_tags[-1]
        v = re.search("(VBN|JJ|NN[PS]*)", head) != None
        v &= re.search("NN[PS]*", tail) != None
        if len(pos_tags) <= 3:
            return v
        v &= len(self.np_tags.intersection(pos_tags[1:-1]))
        return v


# ------------------------------------------------
#
#  NCBI Disease Corpus
#
# ------------------------------------------------

class NcbiDiseaseGenerator(CandidateGenerator):

    def __init__(self, kgrams, dict_only=False,
                 split_chars=['-', '/', '\+'],
                 longest_match_only=True):
        self.kgrams = kgrams
        self.longest_match_only = longest_match_only
        self.dict_only = dict_only
        self.split_chars = split_chars
        self._init_deps()
        self.cand_space = Ngrams(n_max=kgrams, split_tokens=self.split_chars)

    def get_candidates(self, sentences, nprocs=1):
        '''
        Extract candidates
        :param sentences:  sentences to parse
        :param nprocs:  mulit-processing
        :return:
        '''
        cs = Candidates(self.cand_space, self.matcher, sentences, parallelism=nprocs)
        candidates = cs.get_candidates()
        return candidates

    def _init_deps(self):

        DICT_ROOT = "../data/dicts/diseases/snorkel_v0.3/"

        # ==================================================================
        # Prefixes & Suffixes
        # ==================================================================
        # manually created
        inheritance = ["x linked", "x linked recessive", "x linked dominant",
                       "x-linked", "x-linked recessive", "x-linked dominant",
                       "recessive", "dominant", "semidominant", "non-familial",
                       "inherited", "hereditary", "nonhereditary", "familial",
                       "autosomal recessive", "autosomal dominant"]

        alt_inheritance = set(reduce(lambda x, y: x + y, map(lambda x: x.split(), inheritance)))
        alt_inheritance.remove("x")
        alt_inheritance.remove('linked')
        alt_inheritance.add('x linked')

        type_suffix = ['type', 'class', 'stage', 'factor']
        type_nums = ['i', 'ii', 'iii', 'vi', 'v', 'vi', '1a', 'iid', 'a', 'b', 'c', 'd']
        type_nums += map(unicode, range(1, 10))

        nonspecific_diseases = ["disease", "diseases", "syndrome", "syndromes",
                                "disorder", "disorders", "damage", "infection"]

        # ==================================================================
        # The UMLS Semantic Network
        # ==================================================================
        # The UMLS defines 133 fine-grained entity types which are them
        # grouped into coarser semantic categories using groupings defined at:
        #    https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt
        # This set represents semantic types corresponding to "Disorders"
        # with the sub-entity "Finding" removed due to precision issues (10pt drop!)

        disease_entity_types = ["Acquired Abnormality",
                                "Anatomical Abnormality",
                                "Cell or Molecular Dysfunction",
                                "Congenital Abnormality",
                                "Disease or Syndrome",
                                "Experimental Model of Disease",
                                "Injury or Poisoning",
                                "Mental or Behavioral Dysfunction",
                                "Neoplastic Process",
                                "Pathologic Function",
                                "Sign or Symptom"]

        # UMLS terms and abbreviations/acronyms
        umls_disease_terms = UmlsDictionary("terms", sem_types=disease_entity_types)

        # ------------------------------------------------------------------
        # Disease Abbreviations / Acronyms
        # ------------------------------------------------------------------
        umls_disease_abbrvs = UmlsDictionary("abbrvs", sem_types=disease_entity_types)

        umls_specialist_abbrvs = SpecialistLexicon()

        # ------------------------------------------------------------------
        # Stopwords, these are entity types that cause a lot of false positives
        # ------------------------------------------------------------------
        stop_entity_types = ["Geographic Area", "Genetic Function", "Age Group"]
        umls_stopwords = UmlsDictionary("terms", sem_types=stop_entity_types, ignore_case=True)

        # ------------------------------------------------------------------
        # Entities for Slot Filler
        # ------------------------------------------------------------------
        # Molecular Sequence
        entity_types = ["Amino Acid, Peptide, or Protein", "Gene or Genome",
                        "Enzyme", "Nucleic Acid, Nucleoside, or Nucleotide",
                        "Nucleotide Sequence", "Carbohydrate Sequence",
                        "Molecular Sequence"]
        umls_molecular_seq_terms = UmlsDictionary("terms", sem_types=entity_types)

        entity_types = ["Disease or Syndrome"]
        umls_disease_syndrome_terms = UmlsDictionary("terms", sem_types=entity_types, ignore_case=True)

        entity_types = ["Body Part, Organ, or Organ Component"]
        umls_body_part_terms = UmlsDictionary("terms", sem_types=entity_types, ignore_case=True)

        entity_types = ["Neoplastic Process"]
        umls_neoplastic_process_terms = UmlsDictionary("terms", sem_types=entity_types, ignore_case=True)

        entity_types = ["Temporal Concept", "Age Group"]
        umls_temporal_terms = UmlsDictionary("terms", sem_types=entity_types, ignore_case=True)

        entity_types = ["Functional Concept"]
        umls_functional_terms = UmlsDictionary("terms", sem_types=entity_types, ignore_case=True)

        entity_types = ["Finding"]
        umls_finding_terms = UmlsDictionary("terms", sem_types=entity_types, ignore_case=True)

        entity_types = ["Cell"]
        umls_cell_terms = UmlsDictionary("terms", sem_types=entity_types, ignore_case=True)

        # ==================================================================
        # The National Center for Biomedical Ontology
        # http://bioportal.bioontology.org/
        #
        # Comparative Toxicogenomics Database
        # http://ctdbase.org/
        # ==================================================================
        # This uses 4 disease-related ontologies:
        #   (ordo) Orphanet Rare Disease Ontology
        #   (doid) Human Disease Ontology
        #   (hp)   Human Phenotype Ontology
        #   (ctd)  Comparative Toxicogenomics Database

        dict_ordo = load_bioportal_dictionary("{}ordo.csv".format(DICT_ROOT), ignore_case=0)
        dict_doid = load_bioportal_dictionary("{}DOID.csv".format(DICT_ROOT), ignore_case=0)
        dict_hp = load_bioportal_dictionary("{}HP.csv".format(DICT_ROOT), ignore_case=0)
        dict_ctd = load_ctd_dictionary("{}CTD_diseases.tsv".format(DICT_ROOT), ignore_case=0)

        # ==================================================================
        # Manually Created Dictionaries
        # ==================================================================
        # The goal is to minimize this part as much as possible
        # IDEALLY we should build these from the above external curated resources
        # Otherwise these are put together using Wikpipedia and training set debugging

        # Common disease acronyms
        fname = "{}common_disease_acronyms.txt".format(DICT_ROOT)
        dict_common_disease_acronyms = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        fname = "{}stopwords.txt".format(DICT_ROOT)
        dict_stopwords = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        fname = "{}manual_stopwords.txt".format(DICT_ROOT)
        dict_common_stopwords = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        # ------------------------------------------------------------------
        # Disease Terms
        # ------------------------------------------------------------------
        dict_diseases = umls_disease_terms.get_dictionary()
        dict_diseases.update(dict_ordo)
        dict_diseases.update(dict_doid)
        dict_diseases.update(dict_hp)
        dict_diseases.update(dict_ctd)

        # ------------------------------------------------------------------
        # Disease Abbreviations / Acronyms
        # ------------------------------------------------------------------
        dict_disease_abbrvs = umls_disease_abbrvs.get_dictionary()
        # specialist abbreviations kill precision
        # dict_specialist_abbrvs  = umls_specialist_abbrvs.abbrv2text

        # ------------------------------------------------------------------
        # Slot-filled Dictionaries
        # ------------------------------------------------------------------
        dict_molecular_seq = umls_molecular_seq_terms.get_dictionary()
        dict_disease_syndrome = umls_disease_syndrome_terms.get_dictionary()
        dict_body_part = umls_body_part_terms.get_dictionary()
        dict_neoplastic_process = umls_neoplastic_process_terms.get_dictionary()
        dict_temporal = umls_temporal_terms.get_dictionary()
        dict_func_concept = umls_functional_terms.get_dictionary()
        dict_finding = umls_finding_terms.get_dictionary()
        dict_cell = umls_cell_terms.get_dictionary()

        # ------------------------------------------------------------------
        # Stopwords
        # ------------------------------------------------------------------
        dict_stopwords.update(umls_stopwords.get_dictionary())
        dict_stopwords.update(dict.fromkeys(inheritance))

        # MANUAL DICTIONARIES
        # terms
        dict_disease_abbrvs.update(dict_common_disease_acronyms)
        # dict_disease_abbrvs.update(dict_specialist_abbrvs)
        # abbreviations
        dict_stopwords.update(dict_common_stopwords)
        dict_stopwords.update(dict.fromkeys(nonspecific_diseases))

        # ------------------------------------------------------------------
        # Disease Terms
        # ------------------------------------------------------------------
        # strip "disease" and "syndrome" from term string and use some
        # heuristics to filter out false positives like "cardiac"
        dict_disease_roots = get_disease_root_terms(dict_disease_syndrome)

        # Some phrases begin with PRPs like "As If Personality", "of male prostate cancer"
        dict_stop_phrases = {t: 1 for t in dict_diseases if re.search("^([Oo]f|[Aa]s|[Ww]ith) ", t)}
        dict_stopwords.update(dict_stop_phrases)

        # ------------------------------------------------------------------
        # Disease Abbreviations / Acronyms
        # ------------------------------------------------------------------
        # The source vocabulary SNOMEDCT_US encodes abbreviations directly in the
        # source string, so we extract an additional abbreviation dictionary
        dict_snomedct_abbrvs = get_snomedct_disease_abbrvs()

        # Include disease name matches that are uppercase letters, but less than 5 chars.
        # This slightly improves our abbrv/acronym candidate recall
        dict_heuristic_abbrvs = {t: 1 for t in dict_diseases if t.isupper() and len(t) < 5}

        # ==================================================================
        # Update dictionaries
        # ==================================================================
        dict_diseases.update(dict_disease_roots)
        # disease SNOMEDCT fullnames
        tmp = reduce(lambda x, y: x + y, dict_snomedct_abbrvs.values())
        dict_diseases.update(dict.fromkeys(tmp))

        dict_disease_abbrvs.update(dict.fromkeys(dict_snomedct_abbrvs.keys()))
        dict_disease_abbrvs.update(dict_heuristic_abbrvs)

        # ==================================================================
        # FIXES: Parentheticals that are defined in-dictionary
        # ==================================================================
        # By annotation guidelines, mentions of the form:
        # "duchenne muscular dystrophy (DMD)" are two mentions. However,
        # these terms are sometimes present in concatenated forms in our
        # dictionaries, so we extract those term/abbrv pairs and remove them
        ex_abbrv2text = {}
        rm = {}
        for t in dict_diseases:
            if re.search(".+ \([A-Z]{2,}\)$", t):
                m = re.search("^(.+) \(([A-Z]{2,})\)$", t)
                term, abbrv = m.group(1).strip(), m.group(2).strip()
                ex_abbrv2text[abbrv] = ex_abbrv2text.get(abbrv, []) + [term]
                rm[t] = 1

        # scrub terms
        dict_diseases = {t: 1 for t in dict_diseases if t not in rm}
        rm = {x.lower(): 1 for x in rm}
        dict_disease_syndrome = {t: 1 for t in dict_disease_syndrome if t not in rm}

        # Non abbreviation/acronyms like
        # * Registry Nomenclature Information System (RENI)
        stop_abbrvs = ["RENI", "BS", "ZELLWEGER"]

        # add cleaned terms back
        for ab in ex_abbrv2text:
            dict_diseases.update(dict.fromkeys(ex_abbrv2text[ab]))
            dict_disease_syndrome.update(dict.fromkeys([t.lower() for t in ex_abbrv2text[ab]]))
            if ab not in stop_abbrvs:
                dict_disease_abbrvs[ab] = 1

        # Make lowercase
        dict_diseases = {t.lower() for t in dict_diseases}
        dict_disease_roots = {t.lower() for t in dict_disease_roots}

        # GENERAL NOTES: There are some annotation inconsistences:
        # "early-onset" is an incorrect prefix, but "sporadic" is not
        # sometimes "hereditary", "autosomal dominant", etc is a valid prefix; sometimes not

        # Filter out stopwords, single char terms, and digits
        dict_diseases = filter_dictionary(dict_diseases, dict_stopwords)
        dict_disease_abbrvs = filter_dictionary(dict_disease_abbrvs, dict_stopwords)

        dict_molecular_seq = filter_dictionary(dict_molecular_seq, dict_stopwords)
        dict_disease_syndrome = filter_dictionary(dict_disease_syndrome, dict_stopwords)
        dict_body_part = filter_dictionary(dict_body_part, dict_stopwords)
        dict_neoplastic_process = filter_dictionary(dict_neoplastic_process, dict_stopwords)
        dict_temporal = filter_dictionary(dict_temporal, dict_stopwords)
        dict_func_concept = filter_dictionary(dict_func_concept, dict_stopwords)
        dict_finding = filter_dictionary(dict_finding, dict_stopwords)
        dict_cell = filter_dictionary(dict_cell, dict_stopwords)

        # Manual Tweaks
        dict_cell.remove("cell")


        # ==================================================================
        #
        # Matchers
        #
        # ==================================================================

        # Splitting Ngram candidate space on [-/] allows use to match mentions of the form:
        # 10677309   breast cancer-susceptibility
        # 10712209   prostate cancer-susceptibility
        # 8364574    aniridia-associated
        # 7652577    tumor-specific
        ngrams = Ngrams(n_max=self.kgrams)
        longest_match_only = True

        # ------------------------------------------------------------------
        # Matchers: Dictionaries
        # ------------------------------------------------------------------
        MF_dict_diseases = DictionaryMatch(d=dict_diseases, ignore_case=True,
                                           longest_match_only=self.longest_match_only)
        MF_dict_abbrvs = DictionaryMatch(d=dict_disease_abbrvs, ignore_case=False,
                                         longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Disease Root Terms
        # ------------------------------------------------------------------
        # strip disease/syndrome from dictionary terms
        # (e.g., Lou Gehrig's Disease --> Lou Gehrig's)
        # for this matcher, it's useful to keep nonspecific disease terms

        # dict_roots = dict_diseases.union(dict_disease_abbrvs)
        dict_roots = {t for t in dict_diseases}
        dict_roots = dict_roots.union(set(nonspecific_diseases))
        MF_disease_roots = DictionaryMatch(d=dict_roots, ignore_case=True,
                                           longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Slot-filled UMLS Entity Dictionaries
        # ------------------------------------------------------------------
        # Gene/Genome, Protein, other molecular sequences
        MF_molecular_seq = DictionaryMatch(d=dict_molecular_seq,
                                           longest_match_only=self.longest_match_only)
        # Human anatomy
        MF_body_part = DictionaryMatch(d=dict_body_part,
                                       longest_match_only=self.longest_match_only)
        # Cancers
        MF_neoplastic_process = DictionaryMatch(d=dict_neoplastic_process,
                                                longest_match_only=self.longest_match_only)
        # Temporal modifiers
        MF_temporal = DictionaryMatch(d=dict_temporal, longest_match_only=self.longest_match_only)

        # Genetic inheritence modifiers
        MF_inheritance = DictionaryMatch(d=inheritance, longest_match_only=self.longest_match_only)

        # Disease or Syndrome
        MF_disease_syndrome = DictionaryMatch(d=dict_disease_syndrome,
                                              longest_match_only=self.longest_match_only)
        # Cell
        MF_cell = DictionaryMatch(d=dict_cell, longest_match_only=self.longest_match_only)

        # Functional Concept
        MF_func_concept = DictionaryMatch(d=dict_func_concept,
                                          longest_match_only=self.longest_match_only)
        # Finding
        MF_finding = DictionaryMatch(d=dict_finding, longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Typed Diseases
        # ------------------------------------------------------------------
        # 10406661 type II collagenopathy
        # 8113388  type I protein S deficiency
        # 10807385 stage III cancers
        # 8622978  type IID von Willebrand disease  *** BUG ***
        #
        # 8113388  protein S deficiency type I
        # 8571951  Atelosteogenesis type II

        MF_types = Concat(DictionaryMatch(d=type_suffix),
                          DictionaryMatch(d=type_nums))

        MF_disease_types_left = Concat(MF_types, MF_disease_roots)
        MF_disease_types_right = Concat(MF_disease_roots, MF_types)

        # abbreviation typed diseases
        MF_disease_abbrvs_types_left = Concat(MF_types, MF_dict_abbrvs)
        MF_disease_abbrvs_types_right = Concat(MF_dict_abbrvs, MF_types)

        # ------------------------------------------------------------------
        # Matchers: Inherited Diseases
        # ------------------------------------------------------------------
        # 8301658   X linked recessive thrombocytopenia
        # 10577908  autosomal recessive disorder
        # 7550349   inherited breast cancer
        # 10699184  inherited neuromuscular disease
        # 8301658   X linked recessive thrombocytopenia
        #
        # Note: There are inconsistencies in the NCBI corpus where considering
        # inherited terms ("X-linked") as part of the full disease mention.

        MF_genetic_disorders = Concat(MF_inheritance, MF_disease_roots,
                                      longest_match_only=self.longest_match_only)

        # MODIFIED MATCHER
        rgx = "^(({})\s*)+".format("|".join(alt_inheritance).replace("-", "\-"))
        MF_alt_genetic_disorders = Concat(RegexMatchSpan(rgx=rgx, ignore_case=True),
                                          MF_disease_roots,
                                          longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Diseases of Deficiency
        # ------------------------------------------------------------------
        # 1577763  C2-deficient
        # 10429004 PAH-deficient
        suffix_terms = ['deficiency', 'deficient', 'deficienty']
        regex = "^[-]*({})$".format("|".join(suffix_terms))
        MF_deficient = RegexMatchEach(rgx=regex)
        MF_deficient_suffix = Concat(MF_disease_roots, MF_deficient)

        # Pattern: deficiency of [ENZYME|PROTEIN]
        # deficiency of [hepatic fructokinase]
        # 6103091 deficiency of C3
        MF_prefix_terms = DictionaryMatch(d=['deficiency of'])
        MF_deficiency_of = Concat(MF_prefix_terms, MF_disease_roots,
                                  longest_match_only=self.longest_match_only)

        rgx = "^(?!Atm|No|no|non|this|from|the|an|is|of)[A-Z][A-Za-z0-9]{1,4}[- ]"
        rgx += "([Dd]eficient|[Dd]eficiency)$"
        MF_deficient_suffix_regex_type = Concat(MF_types,
                                                RegexMatchSpan(rgx=rgx, ignore_case=False),
                                                left_required=False)

        MF_deficient_suffix_regex_inherited = Concat(MF_inheritance,
                                                     RegexMatchSpan(rgx=regex, ignore_case=False),
                                                     left_required=False)

        # ------------------------------------------------------------------
        # Matchers: Composite Cancers
        # ------------------------------------------------------------------
        # 10788334 breast or ovarian cancer
        # 10441573 breast and/or ovarian cancer
        # 7759075 breast/ovarian cancer
        rgx = "^(advanced|advanced-stage|metastatic) [A-Za-z]+ (cancer|carcinom)[s]*$"
        MF_adj_cancer = RegexMatchSpan(rgx=rgx)

        rgx = "^(?!(cancer|of|or|as|if|the))([A-Za-z])+"
        rgx += "( (and|/|or)+ )(([A-Za-z])+ ){1,2}(cancer|carcinom)[s]*$"
        MF_composite_cancer = RegexMatchSpan(rgx=rgx)

        # 7759075   breast/ovarian cancer
        # 8554067   breast-ovarian cancer
        rgx = "^(?!(cancer|of|or|as|if|the))([A-Za-z])+"
        rgx += "[-/](([A-Za-z])+ ){1,1}(cancer|carcinom)[s]*$"
        MF_composite_hyphen_cancer = Concat(MF_inheritance, RegexMatchSpan(rgx=rgx),
                                            left_required=False)

        # -- NOT INCLUDED --
        # 10788334 breast-ovarian cancer syndrome
        rgx = "(syndrome|disease)[s]*$"
        MF_syndromes = RegexMatchSpan(rgx=rgx)
        MF_hypen_body_parts = SlotFillMatch(MF_body_part, MF_body_part, pattern="{0}-{1}")
        MF_body_part_cancer_or_syndrome = Concat(MF_hypen_body_parts,
                                                 Concat(MF_neoplastic_process, MF_syndromes,
                                                        left_required=False),
                                                 longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Neoplastic Processes
        # ------------------------------------------------------------------
        # body part neoplastic process
        MF_tumor_location = Concat(MF_body_part, MF_neoplastic_process)
        MF_temporal_tumor_location = Concat(MF_temporal, MF_tumor_location, left_required=False)

        # neoplastic tumor
        MF_tumor = RegexMatchSpan(rgx="^(tumor|tumour|carcinoma|cancer)[s]*$")
        MF_neoplastic_tumor = Concat(MF_neoplastic_process, MF_tumor)

        # ------------------------------------------------------------------
        # Matchers: Paranthetical Mentions
        # ------------------------------------------------------------------
        # 10554035    von Hippel-Lindau (VHL) disease
        # 1999339     Glucose-6-phosphate dehydrogenase (G6PD) deficiency
        # 10924409    adenomatous polyposis coli (APC) tumor
        # 2601691     retinoblastoma (RB) tumor
        # 2352258     Von Hippel-Lindau (VHL) disease
        MF_parenthetical_disease = Concat(Concat(MF_disease_roots,
                                                 RegexMatchSpan(rgx="^\(.+\)$")),
                                          RegexMatchSpan(rgx="^(disease|syndrome|disorder)[s]*$"))

        # 1999339  Glucose-6-phosphate dehydrogenase (G6PD) deficiency
        MF_parenthetical_difficiency = Concat(Concat(MF_molecular_seq,
                                                     RegexMatchSpan(rgx="^\(.+\)$")),
                                              RegexMatchSpan(rgx="^(deficiency|\-deficient|deficiencies)$"))

        # 10982189  adenomatous polyposis coli (APC) tumor
        MF_parenthetical_tumor = Concat(Concat(MF_neoplastic_process,
                                               RegexMatchSpan(rgx="^\(.+\)$")),
                                        MF_tumor)

        # ------------------------------------------------------------------
        # Matchers: Disease Location
        # ------------------------------------------------------------------
        MF_disease_location = Concat(MF_temporal, Concat(MF_body_part, MF_disease_syndrome),
                                     left_required=False)

        # (genetic - (temporal - (body part - disease)))
        MF_inherited_disease_location = Concat(MF_inheritance,
                                               MF_disease_location,
                                               left_required=False)

        # ------------------------------------------------------------------
        # Matchers: Syndromes
        # ------------------------------------------------------------------
        rgx = "^(syndromic|non\-syndromic)$"
        MF_syndromes = RegexMatchSpan(rgx=rgx, ignore_case=True)

        MF_syndromic_disease = Concat(MF_syndromes, MF_disease_syndrome,
                                      longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Deficiency of * (Noun Phrase)
        # ------------------------------------------------------------------
        rgx = "^(DT )*((NN[SP]*|IN|CC|JJ|[:,+])\s*)*(NN[SP]*)$"
        MF_np_matcher = RegexMatchSpan(attrib="poses", rgx=rgx, ignore_case=True,
                                       longest_match_only=self.longest_match_only)

        rgx = "^((inherited|familial|hereditary|congenital) )*deficiency of$"
        MF_def_of_matcher = RegexMatchSpan(rgx=rgx, ignore_case=True)

        MF_deficiency_noun_phrase = Concat(MF_def_of_matcher, MF_np_matcher)

        # ------------------------------------------------------------------
        # Matchers: Abnormalities
        # ------------------------------------------------------------------
        # 3258663   genetic abnormality
        # 3258663   genetic abnormalities
        # 6783144   haemostasis abnormality
        rgx = "^(?!(the|to|only|an|a))[A-Za-z]{2,} (abnormalit(y|ies)|anomal(y|ies)|defect[s]*)$"
        MF_abnormalities = RegexMatchSpan(rgx=rgx, ignore_case=True)

        # ------------------------------------------------------------------
        # Matchers: Cell Neoplastic Processes
        # ------------------------------------------------------------------
        # 10830910    renal cell carcinoma
        MF_cell_neoplastic_process = Concat(MF_cell, MF_neoplastic_process)
        MF_temporal_cell_neoplastic_process = Concat(MF_temporal, MF_cell_neoplastic_process,
                                                     left_required=False)

        # ------------------------------------------------------------------
        # Slot Filled Matchers: Genetic Abnormality
        # ------------------------------------------------------------------
        # TODO: sub-character spans
        # MF_slot_abbrvs = SlotFillMatch(dict_abbrvs, dict_abbrvs, pattern="{0}/{1}")
        # 7481765    BRCA1 abnormalities
        MF_genetic_abnormalities = SlotFillMatch(MF_molecular_seq,
                                                 RegexMatchSpan(rgx="abnormalit(ies|y)"),
                                                 pattern="{0} {1}",
                                                 longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Idiopathic
        # ------------------------------------------------------------------
        # 8528198   isolated thrombocytopenia
        # 10737119  idiopathic torsion dystonia
        rgx = "^(idiopathic|isolated)"
        MF_idiopathic = RegexMatchSpan(rgx=rgx, ignore_case=True)
        MF_idiopathic_disease = Concat(MF_idiopathic, MF_disease_syndrome,
                                       longest_match_only=self.longest_match_only)

        # ------------------------------------------------------------------
        # Matchers: Abbreviations / Acronyms
        # ------------------------------------------------------------------
        # acronyms, filtered for gene sequences, high freq. gene names, roman numerals, and other false positives
        # from training set; any term in the UMLS that is not a disease/disorder semantic type is removed
        rm_abrv = ["VLCFA", "SNRPN", "HEXA", "PEPCK", "ALDRP", "WASP", "HPRT", "CETP", "BRCA", "FMRP", "LCAT", "ALDP",
                   "SSCP", "COS",
                   "PLP", "OTC", "DNA", "PCR", "RNA", "WHO", "ATP", "GTP", "DDP", "HDL", "LDL", "HGD", "HLA", "GTD",
                   "BMT", "GK"]
        rgx = "^(?!([ATCG]{2,4}|[IX]{1,3}|" + "|".join(rm_abrv) + ")$)[A-Z]{2,5}$"
        MF_acronym_abbreviation = RegexMatchSpan(rgx=rgx, ignore_case=False)

        fuzzy_matcher = Union(
            MF_disease_types_left,
            MF_disease_types_right,
            MF_disease_abbrvs_types_left,
            MF_disease_abbrvs_types_right,

            Concat(MF_disease_types_left, MF_deficient),
            MF_deficient_suffix,
            MF_deficiency_of,
            MF_genetic_disorders,
            Concat(MF_inheritance, MF_composite_cancer),
            MF_composite_cancer,
            MF_adj_cancer,
            MF_composite_hyphen_cancer,

            MF_disease_roots,
            MF_parenthetical_disease,
            MF_parenthetical_difficiency,
            MF_parenthetical_tumor,
            MF_temporal_cell_neoplastic_process,
            MF_temporal_tumor_location,

            MF_deficient_suffix_regex_type,
            MF_deficient_suffix_regex_inherited,
            MF_deficiency_noun_phrase,
            MF_abnormalities,
            MF_neoplastic_tumor,
            MF_syndromic_disease,
            MF_genetic_abnormalities,
            MF_idiopathic_disease,

            MF_acronym_abbreviation,

            MF_dict_diseases,
            MF_dict_abbrvs,

            longest_match_only=self.longest_match_only
        )

        dict_matcher = Union(
            MF_dict_diseases,
            MF_dict_abbrvs,
            longest_match_only=self.longest_match_only
        )

        self.matcher = fuzzy_matcher if not self.dict_only else dict_matcher

# ------------------------------------------------
#
#  CDR Chemicals
#
# ------------------------------------------------
class CdrChemicalDictGenerator(CandidateGenerator):
    '''
     @authors Jason Fries & Sen Wu
    '''

    def __init__(self, kgrams, split_chars=['-', '/', '\+'],
                 stopwords=[], longest_match_only=True):
        '''

        :param kgrams:
        :param split_chars:
        :param stopwords:
        :param longest_match_only:
        '''
        super(CdrChemicalDictGenerator, self).__init__(kgrams, split_chars)
        self.stopwords = stopwords
        self._init_deps()

        dict_chemicals = DictionaryMatch(d=self.chemicals, ignore_case=True,
                                         longest_match_only=longest_match_only)
        dict_acronyms = DictionaryMatch(d=self.chemical_acronyms, ignore_case=False,
                                        longest_match_only=longest_match_only)

        self.matcher = Union(
            dict_chemicals,
            dict_acronyms,
            longest_match_only=longest_match_only)

        self.cand_space = Ngrams(n_max=self.kgrams, split_tokens=self.split_chars)

    def get_candidates(self, sentences, nprocs=1):
        cs = Candidates(self.cand_space, self.matcher, sentences, parallelism=nprocs)
        return cs.get_candidates()

    def _init_deps(self):
        '''
        Initialize external resource (e.g., dictionary) dependencies
        for chemical name extraction.
        '''
        #
        # 1. Load UMLS dictionaries
        #
        entity_types = ['Antibiotic', 'Carbohydrate', 'Chemical', 'Eicosanoid',
                        'Element, Ion, or Isotope', 'Hazardous or Poisonous Substance',
                        'Indicator, Reagent, or Diagnostic Aid', 'Inorganic Chemical',
                        'Neuroreactive Substance or Biogenic Amine', 'Nucleic Acid, Nucleoside, or Nucleotide',
                        'Organic Chemical', 'Organophosphorus Compound', 'Steroid', 'Vitamin', 'Lipid']

        umls_terms = UmlsNoiseAwareDict(positive=entity_types, name="terms", ignore_case=False)
        umls_abbrv = UmlsNoiseAwareDict(positive=entity_types, name="abbrvs", ignore_case=False)

        self.chemicals = umls_terms.dictionary()
        self.chemical_acronyms = umls_abbrv.dictionary()


class CdrChemicalGenerator(CandidateGenerator):
    '''
    @authors Jason Fries & Sen Wu
    '''
    def __init__(self, kgrams, clusters,
                 split_chars=['-', '/', '\+'],
                 stopwords=[], longest_match_only = True):
        '''

        :param kgrams:
        :param clusters:
        :param split_chars:
        :param stopwords:
        :param longest_match_only:
        '''
        super(CdrChemicalGenerator, self).__init__(kgrams, split_chars)
        self.stopwords = stopwords
        self.clusters = clusters
        self._init_deps()

        dict_chemicals = DictionaryMatch(d=self.chemicals, ignore_case=True,
                                         longest_match_only=longest_match_only)
        dict_acronyms = DictionaryMatch(d=self.chemical_acronyms, ignore_case=False,
                                        longest_match_only=longest_match_only)

        rgx_matcher = RegexMatchSpan(rgx="^([A-Za-z0-9\[\]():;+'.,-]+\s*){1," + str(kgrams) + "}$",
                                     ignore_case=True)

        if self.clusters != None:
            dict_knn_chemicals = DictionaryMatch(d=self.chemicals_knn, ignore_case=False,
                                            longest_match_only=longest_match_only)
            self.matcher = Union(
                dict_chemicals,
                dict_acronyms,
                dict_knn_chemicals,
                #rgx_matcher,
                longest_match_only=longest_match_only)
        else:
            self.matcher = Union(
                dict_chemicals,
                dict_acronyms,
                #rgx_matcher,
                longest_match_only=longest_match_only)

        self.cand_space = Ngrams(n_max=self.kgrams, split_tokens=self.split_chars)


    def get_candidates(self, sentences, nprocs=1):

        cs = Candidates(self.cand_space, self.matcher, sentences, parallelism=nprocs)
        return cs.get_candidates()


    def _init_deps(self):
        '''
        Initialize external resource (e.g., dictionary) dependencies
        for chemical name extraction. This includes custom stopwords lists
        and other manual dictionary curation.
        '''
        #
        # 1. Load UMLS dictionaries
        #
        entity_types = ['Antibiotic', 'Carbohydrate', 'Chemical', 'Eicosanoid',
                        'Element, Ion, or Isotope', 'Hazardous or Poisonous Substance',
                        'Indicator, Reagent, or Diagnostic Aid', 'Inorganic Chemical',
                        'Neuroreactive Substance or Biogenic Amine', 'Nucleic Acid, Nucleoside, or Nucleotide',
                        'Organic Chemical', 'Organophosphorus Compound', 'Steroid', 'Vitamin', 'Lipid']

        umls_terms = UmlsNoiseAwareDict(positive=entity_types, name="terms", ignore_case=False)
        umls_abbrv = UmlsNoiseAwareDict(positive=entity_types, name="abbrvs", ignore_case=False)

        self.chemicals = umls_terms.dictionary()
        self.chemical_acronyms = umls_abbrv.dictionary()

        #
        # 2. Custom stopwords from error analysis
        #
        diseases = ["pain", "hypertension", "hypertensive", "depression",
                    "depressive", "depressed", "bleeding", "infection",
                    "poisoning", "anxiety", "deaths", "startle"]

        sw = ['acid', 'active', 'add', 'adrenoceptor', 'advantage', 'analgesic', 'anesthetic',
              'animal', 'animals', 'anti-inflammatory', 'antiarrhythmic', 'antibody', 'anticonvulsant',
              'antiepileptic', 'antioxidant', 'baseline', 'basis', 'block', 'blockade',
              'central nervous system', 'chemical', 'combinations', 'complex', 'component',
              'compound', 'conclude', 'contrast', 'control', 'diagnostic', 'dopaminergic', 'drug',
              'drugs', 'element', 'elements', 'food', 'glucocorticoid', 'glucocorticoids', 'hemoglobin',
              'hepatitis', 'hepatitis b', 'hg', 'immunosuppressive', 'inhibitors', 'injection', 'label',
              'labeled', 'level', 'lipid', 'lipids', 'mediated', 'medication', 'medications',
              'metabolites', 'monitor', 'mouse', 'nervous system', 'neuronal', 'opioid', 'oral',
              'placebo', 'pressor', 'prevent', 'prolactin', 'prophylactic', 'purpose', 'related',
              'retinal', 's-1', 'salt', 'sham', 'smoke', 'solution', 'stopping', 'stress', 'support',
              'task', 'therapeutic', 'thyroid', 'today', 'tonic', 'topical', 'transcript', 'triad',
              'unknown', 'various', 'vessel', 'vitamin', 'water']
        sw += ["normal saline", "lead", "convulsant", "antihypertensive", "stopping", "chemical",
               "label", "antibiotics", "pesticide", "sham", "anticoagulant", "antimicrobial",
               "aprotinin", "ligands", "pituitary", "neurotransmitter", "neurotransmitters"]

        # non_common_chemical_acronyms
        sw += ['AR', 'As', 'CR', 'CRF', 'CT', 'DA', 'DS', 'ECG', 'FU', 'GH', 'HDL', 'He', 'IGF', 'IP',
               'LH', 'LV', 'M4', 'MB', 'MPO', 'NF', 'NSAID', 'NSAIDs', 'OH', 'PAF', 'PRL', 'TBPS',
               'TG', 'TMA', 'TMP', 'VIP', 'VTE', 'mtDNA']
        sw += ["V", "IV", "III", "II", "I", "cm", "mg", "pH", "In", "Hg", "VIP"]

        self.stopwords.update(dict.fromkeys(diseases))
        self.stopwords.update(dict.fromkeys(sw))

        #
        # 3. Additional term filtering from error analysis
        #
        self.chemicals = {t.lower().strip(): 1 for t in self.chemicals if
                     t.lower().strip() not in [i.lower() for i in self.stopwords.keys()] and len(t) > 1}
        self.chemical_acronyms = {t.strip(): 1 for t in self.chemical_acronyms if t.strip() not in self.stopwords and len(t) > 1}

        ban = ['agent', 'agonist', 'animals', 'antagonist', 'baseline', 'channel', 'control',
               'drug', 'duration', 'fat', 'injection', 'isotonic', 'kinase', 'level', 'liposomal',
               'monohydrate', 'oil', 'pain', 'phosphokinase', 'receptor', 'red', 'related', 'speed',
               'stress', 'system', 'total', 'transporter', 'vehicle', 'yellow']

        # filter out some noisy dictionary matches
        for phrase in self.chemicals.keys():
            check = False
            for i in ban:
                if i.lower() in phrase.lower():
                    check = True
                    break
            if len(phrase) < 3 and phrase.islower():
                check = True
            if phrase.endswith('ic'):
                check = True
            if phrase.endswith('+'):
                check = True
            a = phrase.lower().split()
            if len(a) == 2 and a[0].isdigit() and a[1] == 'h':
                check = True
                #     if len(phrase)>2 and phrase[0].isalpha() and phrase[1]=='-' and phrase[2:].isdigit():
                #         check=True
                #         print phrase

            if check:
                del self.chemicals[phrase]

        self.chemical_acronyms.update(
            dict.fromkeys(["CaCl(2)", "PAN", "SNP", "K", "AX", "VPA", "PG-9", "SRL", "ISO", "CAA", "CBZ", "CPA",
                           "GEM", "CY", "OC", "OCs", "Ca", "PTZ", "NMDA", "H2O", "CsA", "DA", "GSH", "HBsAg", "Rg1"]))

        self.chemicals.update(
            dict.fromkeys(["glutamate", "aspartate", "creatine", "angiotensin", "glutathione", "srl", "dex", "tac",
                           "cya", "l-dopa", "hbeag", "argatroban", "melphalan", "cyclosporine", "enalapril",
                           "l-arginine", "vasopressin", "cyclosporin a", "n-methyl-d-aspartate", "ace inhibitor",
                           "oral contraceptives", "l-name", "alanine", "amino acid", "lisinopril", "tyrosine",
                           "fenfluramines", "beta-carboline", "glutamine", "octreotide", "angiotensin II"]))
        #
        # 4. Augment with cluster terms
        #
        if self.clusters != None:
            cid2words = defaultdict(list)
            for term,cid in self.clusters.items():
                cid2words[cid].append(term)

            self.cids = defaultdict(int)

            for term in self.chemicals:
                tokens = term.split()
                if len(tokens) == 1 and term in self.clusters:
                    cid = self.clusters[term]
                    self.cids[cid] += 1

            for term in self.chemical_acronyms:
                if term in self.clusters:
                    cid = self.clusters[term]
                    self.cids[cid] += 1

            expanded_dict = {}
            for cid,freq in sorted(self.cids.items(), key=lambda x: x[1], reverse=1):
                if freq < 100:
                    break

                expanded_dict.update(dict.fromkeys(cid2words[cid]))

            self.chemicals_knn = expanded_dict

        print "CdrChemicalGenerator initialized..."


# ------------------------------------------------
#
#  CDR Disease
#
# ------------------------------------------------
class CdrDiseaseDictGenerator(CandidateGenerator):
    '''
    @authors Jason Fries & Sen Wu
    '''
    def __init__(self, kgrams, clusters,
                 split_chars=['-', '/', '\+'],
                 stopwords=[], longest_match_only = True):

        super(CdrDiseaseDictGenerator, self).__init__(kgrams, split_chars)
        self.stopwords = stopwords
        self.clusters = clusters
        self._init_deps()

        self.cand_space = Ngrams(n_max=self.kgrams, split_tokens=self.split_chars)


    def get_candidates(self, sentences, nprocs=1):

        cs = Candidates(self.cand_space, self.matcher, sentences, parallelism=nprocs)
        return cs.get_candidates()


    def _init_deps(self):

        DICT_ROOT = "../data/dicts/diseases/snorkel_v0.3/"

        # ==================================================================
        # Domain-specific Stopwords
        # ==================================================================
        disease_stopwords = ["disease", "diseases", "syndromes", "deficiency", "infections",
                             "syndrome", "disorder", "disorders", "damage", "infection",
                             "deficient", "defect", "infectious", "deficit", "acute", "chronic",
                             "all ages", "adult onset", "late onset", "worldwide"]

        disease_stopwords += ['infarction', 'complication', 'impairment', 'adverse effects', 'pressure',
                              'hemorrhage', 'block', 'pathogenesis', 'adverse effect', 'suppression',
                              'sensitization', 'haemorrhage', 'injection', 'symptoms', 'anaesthesia',
                              'anesthesia', 'symptom', 'central nervous system', 'hepatic', 'respiratory',
                              'conditions', 'idiopathic', 'blood', 'mole', 'proliferation', 'occlusion',
                              'infiltrate', 'separation', 'bleeding', 'exposure', 'injury', 'muscle',
                              'strains', 'strain', 'exposures', 'mug', 'prick', 'damages', 'fracture', 'assaults',
                              "side effect", "disease progression", "adverse", 'case', 'prevalence']

        disease_stopwords += ['severe', 'central', 'severityonset', 'severity', 'onset', 'peripheral', 'relapse',
                              'mild', 'progressive', 'moderate', 'moderate', 'distal', 'localized', 'emergency',
                              'intensity', 'right', 'mitochondrial', 'left', 'focal', 'bilateral', 'recurrence',
                              'acute onset', 'diffuse', 'unilateral', 'left-sided', 'localised', 'proximal']

        stop_entity_types = ["Geographic Area", "Genetic Function"]
        umls_stop_terms = UmlsNoiseAwareDict(positive=stop_entity_types,
                                             name="terms", ignore_case=True)

        disease_stopwords += umls_stop_terms.dictionary(min_size=0).keys()
        disease_stopwords = set(disease_stopwords)

        # ==================================================================
        # Prefixes & Suffixes
        # ==================================================================
        # manually created
        inheritance = ["x linked", "x linked recessive", "x linked dominant",
                       "x-linked", "x-linked recessive", "x-linked dominant",
                       "recessive", "dominant", "semidominant", "non-familial",
                       "inherited", "hereditary", "nonhereditary", "familial",
                       "autosomal recessive", "autosomal dominant"]

        # ==================================================================
        # The UMLS Semantic Network
        # ==================================================================
        # The UMLS defines 133 fine-grained entity types which are them
        # grouped into coarser semantic categories using groupings defined at:
        #    https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt
        # This set represents semantic types corresponding to "Disorders"
        # with the sub-entity "Finding" removed due to precision issues (10pt drop!)

        disease_entity_types = ["Acquired Abnormality",
                                "Anatomical Abnormality",
                                "Cell or Molecular Dysfunction",
                                "Congenital Abnormality",
                                "Disease or Syndrome",
                                "Experimental Model of Disease",
                                "Injury or Poisoning",
                                "Mental or Behavioral Dysfunction",
                                "Neoplastic Process",
                                "Pathologic Function",
                                "Sign or Symptom"]

        # UMLS terms and abbreviations/acronyms
        umls_disease_terms = UmlsNoiseAwareDict(positive=disease_entity_types, rm_sab=["LPN"],
                                                name="terms", ignore_case=False)
        # Disease Abbreviations / Acronyms
        umls_disease_abbrvs = UmlsNoiseAwareDict(positive=disease_entity_types, rm_sab=["LPN"],
                                                 name="abbrvs", ignore_case=False)

        # ==================================================================
        # The National Center for Biomedical Ontology
        # http://bioportal.bioontology.org/
        #
        # Comparative Toxicogenomics Database
        # http://ctdbase.org/
        # ==================================================================
        # This uses 4 disease-related ontologies:
        #   (ordo) Orphanet Rare Disease Ontology
        #   (doid) Human Disease Ontology
        #   (hp)   Human Phenotype Ontology
        #   (ctd)  Comparative Toxicogenomics Database

        dict_ordo = load_bioportal_dictionary("{}ordo.csv".format(DICT_ROOT))
        dict_doid = load_bioportal_dictionary("{}DOID.csv".format(DICT_ROOT))
        dict_hp = load_bioportal_dictionary("{}HP.csv".format(DICT_ROOT))
        dict_ctd = load_ctd_dictionary("{}CTD_diseases.tsv".format(DICT_ROOT))

        dict_ordo = dict.fromkeys(dict_ordo)
        dict_doid = dict.fromkeys(dict_doid)
        dict_hp = dict.fromkeys(dict_hp)
        dict_ctd = dict.fromkeys(dict_ctd)

        # ==================================================================
        # Other Curated Dictionaries
        # ==================================================================
        # The goal is to minimize this part as much as possible
        # IDEALLY we should build these from the above external curated resources
        # Otherwise these are put together using Wikpipedia and training set debugging

        # Common disease acronyms
        fname = "{}common_disease_acronyms.txt".format(DICT_ROOT)
        dict_common_disease_acronyms = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        fname = "{}stopwords.txt".format(DICT_ROOT)
        dict_stopwords = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        fname = "{}manual_stopwords.txt".format(DICT_ROOT)
        dict_common_stopwords = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        diseases = umls_disease_terms.dictionary(min_size=0)
        abbrvs = umls_disease_abbrvs.dictionary(min_size=0)

        # Update with all uppercase terms from disease dictionary
        abbrvs.update({term: 1 for term in diseases if term.isupper() and len(term) > 1})

        diseases.update(dict_ordo)
        diseases.update(dict_doid)
        diseases.update(dict_hp)
        diseases.update(dict_ctd)
        diseases.update(abbrvs)

        # default (non-domain) stopwords
        stopwords = dict_stopwords

        abbrvs.update(dict.fromkeys(dict_common_disease_acronyms))

        # remove stopwords
        diseases = {t.lower().strip(): 1 for t in diseases if t.lower().strip() not in stopwords and len(t) > 1}
        abbrvs = {t: 1 for t in abbrvs if len(t) > 1 and t not in stopwords}


        #
        # DICTIONARIES
        #
        longest_match_only = True
        dict_diseases = DictionaryMatch(d=diseases, ignore_case=True,
                                        longest_match_only=longest_match_only)
        dict_abbrvs = DictionaryMatch(d=abbrvs, ignore_case=False,
                                      longest_match_only=longest_match_only)
        self.matcher = Union(
            dict_diseases,
            dict_abbrvs,
            longest_match_only=longest_match_only)

        print "CdrDiseaseDictGenerator initialized..."


class CdrDiseaseGenerator(CandidateGenerator):
    '''
    @authors Jason Fries & Sen Wu
    '''
    def __init__(self, kgrams, clusters,
                 split_chars=['-', '/', '\+'],
                 stopwords=[], longest_match_only = True):

        super(CdrDiseaseGenerator, self).__init__(kgrams, split_chars)
        self.stopwords = stopwords
        self.clusters = clusters
        self._init_deps()

        self.cand_space = Ngrams(n_max=self.kgrams, split_tokens=self.split_chars)


    def get_candidates(self, sentences, nprocs=1):

        cs = Candidates(self.cand_space, self.matcher, sentences, parallelism=nprocs)
        return cs.get_candidates()


    def _init_deps(self):

        DICT_ROOT = "../data/dicts/diseases/snorkel_v0.3/"

        # ==================================================================
        # Domain-specific Stopwords
        # ==================================================================
        disease_stopwords = ["disease", "diseases", "syndromes", "deficiency", "infections",
                             "syndrome", "disorder", "disorders", "damage", "infection",
                             "deficient", "defect", "infectious", "deficit", "acute", "chronic",
                             "all ages", "adult onset", "late onset", "worldwide"]

        disease_stopwords += ['infarction', 'complication', 'impairment', 'adverse effects', 'pressure',
                              'hemorrhage', 'block', 'pathogenesis', 'adverse effect', 'suppression',
                              'sensitization', 'haemorrhage', 'injection', 'symptoms', 'anaesthesia',
                              'anesthesia', 'symptom', 'central nervous system', 'hepatic', 'respiratory',
                              'conditions', 'idiopathic', 'blood', 'mole', 'proliferation', 'occlusion',
                              'infiltrate', 'separation', 'bleeding', 'exposure', 'injury', 'muscle',
                              'strains', 'strain', 'exposures', 'mug', 'prick', 'damages', 'fracture', 'assaults',
                              "side effect", "disease progression", "adverse", 'case', 'prevalence']

        disease_stopwords += ['severe', 'central', 'severityonset', 'severity', 'onset', 'peripheral', 'relapse',
                              'mild', 'progressive', 'moderate', 'moderate', 'distal', 'localized', 'emergency',
                              'intensity', 'right', 'mitochondrial', 'left', 'focal', 'bilateral', 'recurrence',
                              'acute onset', 'diffuse', 'unilateral', 'left-sided', 'localised', 'proximal']

        stop_entity_types = ["Geographic Area", "Genetic Function"]
        umls_stop_terms = UmlsNoiseAwareDict(positive=stop_entity_types,
                                             name="terms", ignore_case=True)

        disease_stopwords += umls_stop_terms.dictionary(min_size=0).keys()
        disease_stopwords = set(disease_stopwords)

        # ==================================================================
        # Prefixes & Suffixes
        # ==================================================================
        # manually created
        inheritance = ["x linked", "x linked recessive", "x linked dominant",
                       "x-linked", "x-linked recessive", "x-linked dominant",
                       "recessive", "dominant", "semidominant", "non-familial",
                       "inherited", "hereditary", "nonhereditary", "familial",
                       "autosomal recessive", "autosomal dominant"]

        # ==================================================================
        # The UMLS Semantic Network
        # ==================================================================
        # The UMLS defines 133 fine-grained entity types which are them
        # grouped into coarser semantic categories using groupings defined at:
        #    https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt
        # This set represents semantic types corresponding to "Disorders"
        # with the sub-entity "Finding" removed due to precision issues (10pt drop!)

        disease_entity_types = ["Acquired Abnormality",
                                "Anatomical Abnormality",
                                "Cell or Molecular Dysfunction",
                                "Congenital Abnormality",
                                "Disease or Syndrome",
                                "Experimental Model of Disease",
                                "Injury or Poisoning",
                                "Mental or Behavioral Dysfunction",
                                "Neoplastic Process",
                                "Pathologic Function",
                                "Sign or Symptom"]

        # UMLS terms and abbreviations/acronyms
        umls_disease_terms = UmlsNoiseAwareDict(positive=disease_entity_types, rm_sab=["LPN"],
                                                name="terms", ignore_case=False)
        # Disease Abbreviations / Acronyms
        umls_disease_abbrvs = UmlsNoiseAwareDict(positive=disease_entity_types, rm_sab=["LPN"],
                                                 name="abbrvs", ignore_case=False)

        # ==================================================================
        # The National Center for Biomedical Ontology
        # http://bioportal.bioontology.org/
        #
        # Comparative Toxicogenomics Database
        # http://ctdbase.org/
        # ==================================================================
        # This uses 4 disease-related ontologies:
        #   (ordo) Orphanet Rare Disease Ontology
        #   (doid) Human Disease Ontology
        #   (hp)   Human Phenotype Ontology
        #   (ctd)  Comparative Toxicogenomics Database

        dict_ordo = load_bioportal_dictionary("{}ordo.csv".format(DICT_ROOT))
        dict_doid = load_bioportal_dictionary("{}DOID.csv".format(DICT_ROOT))
        dict_hp = load_bioportal_dictionary("{}HP.csv".format(DICT_ROOT))
        dict_ctd = load_ctd_dictionary("{}CTD_diseases.tsv".format(DICT_ROOT))

        dict_ordo = dict.fromkeys(dict_ordo)
        dict_doid = dict.fromkeys(dict_doid)
        dict_hp = dict.fromkeys(dict_hp)
        dict_ctd = dict.fromkeys(dict_ctd)

        # ==================================================================
        # Other Curated Dictionaries
        # ==================================================================
        # The goal is to minimize this part as much as possible
        # IDEALLY we should build these from the above external curated resources
        # Otherwise these are put together using Wikpipedia and training set debugging

        # Common disease acronyms
        fname = "{}common_disease_acronyms.txt".format(DICT_ROOT)
        dict_common_disease_acronyms = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        fname = "{}stopwords.txt".format(DICT_ROOT)
        dict_stopwords = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        fname = "{}manual_stopwords.txt".format(DICT_ROOT)
        dict_common_stopwords = dict.fromkeys([l.strip() for l in open(fname, "rU")])

        diseases = umls_disease_terms.dictionary(min_size=0)
        abbrvs = umls_disease_abbrvs.dictionary(min_size=0)

        # Update with all uppercase terms from disease dictionary
        abbrvs.update({term: 1 for term in diseases if term.isupper() and len(term) > 1})

        diseases.update(dict_ordo)
        diseases.update(dict_doid)
        diseases.update(dict_hp)
        diseases.update(dict_ctd)
        diseases.update(abbrvs)

        # default (non-domain) stopwords
        stopwords = dict_stopwords


        # =======
        # Curate
        # =======
        # use domain-specific stopword lists
        stop_entities = umls_stop_terms.dictionary(min_size=0)
        stopwords.update({term.lower(): 1 for term in stop_entities})
        stopwords.update(dict.fromkeys(dict_common_stopwords))
        disease_or_syndrome = umls_disease_terms.dictionary(semantic_types=["disease_or_syndrome"])
        stopwords.update(get_umls_stopwords(keep_words=disease_or_syndrome))
        stopwords.update(dict.fromkeys(inheritance))

        # add some common abbreviations
        dict_common_disease_acronyms.update(dict.fromkeys(["TNSs", "LIDs", "TDP", "TMA", "TG", "SE", "ALF",
                                                           "CHC", "RPN", "HITT", "VTE", "HEM", "NIN", "LID",
                                                           "CIMD", "MAHA", "LID", "CIMD", "MAHA"]))
        abbrvs.update(dict.fromkeys(dict_common_disease_acronyms))

        diseases = {t.lower().strip(): 1 for t in diseases if t.lower().strip() not in stopwords and len(t) > 1}

        diseases = {t.lower().strip(): 1 for t in diseases if not t.lower().strip().startswith('heparin-induced')}
        abbrvs = {t: 1 for t in abbrvs if len(t) > 1 and t not in stopwords}

        # we have these in the UMLS -- why do they get stripped out?
        diseases.update(dict.fromkeys(["melanoma", "qt prolongation", "seizure", "overdose", "tdp"]))

        # general disease
        diseases.update(dict.fromkeys(["pain", "hypertension", "hypertensive", "depression", "depressive", "depressed",
                                       "bleeding", "infection", "poisoning", "anxiety", "deaths", "startle"]))

        # common disease
        diseases.update(dict.fromkeys(['parkinsonian', 'convulsive', 'leukocyturia', 'bipolar', 'pseudolithiasis',
                                       'malformations', 'angina', 'dysrhythmias', 'calcification', 'paranoid',
                                       'hiv-infected']))

        # adj disease
        diseases.update(
            dict.fromkeys(['acromegalic', 'akinetic', 'allergic', 'arrhythmic', 'arteriopathic', 'asthmatic',
                           'atherosclerotic', 'bradycardic', 'cardiotoxic', 'cataleptic', 'cholestatic',
                           'cirrhotic', 'diabetic', 'dyskinetic', 'dystonic', 'eosinophilic', 'epileptic',
                           'exencephalic', 'haemorrhagic', 'hemolytic', 'hemorrhagic', 'hemosiderotic', 'hepatotoxic'
                                                                                                        'hyperalgesic',
                           'hyperammonemic', 'hypercalcemic', 'hypercapnic', 'hyperemic',
                           'hyperkinetic', 'hypertrophic', 'hypomanic', 'hypothermic', 'ischaemic', 'ischemic',
                           'leukemic', 'myelodysplastic', 'myopathic', 'necrotic', 'nephrotic', 'nephrotoxic',
                           'neuropathic', 'neurotoxic', 'neutropenic', 'ototoxic', 'polyuric', 'proteinuric',
                           'psoriatic', 'psychiatric', 'psychotic', 'quadriplegic', 'schizophrenic', 'teratogenic',
                           'thromboembolic', 'thrombotic', 'traumatic', 'vasculitic']))

        # remove disease name like chronic renal failure (CRF)
        for d in diseases.keys():
            if d[-1] == ')':
                s = d.split(' (')
                if len(s) == 2:
                    s1, s2 = s[0], s[1][:-1]
                    s3 = ''.join([i[0] for i in s1.replace('-', ' ').split()]).lower()
                    if s2.lower() == s3.lower():
                        del diseases[d]
                        diseases[s1] = 1
                        abbrvs[s2.upper()] = 1
                    s3 = ''.join([i[0] for i in s1.split()]).lower()
                    if s2.lower() == s3.lower() and d in diseases:
                        del diseases[d]
                        diseases[s1] = 1
                        abbrvs[s2.upper()] = 1

        # remove disease name contains general term
        for d in diseases.keys():
            check = False
            if d.lower().split()[0] in ['generalized', 'metastatic', 'recurrent', 'complete', 'immune', 'deficit']:
                check = True
            if len(d.lower().split()[0]) == 1:
                check = True
            if d.lower().split()[-1] in ['event', 'events', 'effect', 'effects']:
                check = True
            if d.lower()[0] == 'p' and d.lower()[1:].isdigit():
                check = True
            if re.match("^\d+?\.\d+?$", d) is not None or d.isdigit():
                check = True
            if d.islower() and len(d) < 4:
                check = True
            if check:
                del diseases[d]


        # ==================================================================
        # Matchers
        # ==================================================================

        #
        # DICTIONARIES
        #
        longest_match_only = True
        stemmer = 'porter'
        dict_diseases = DictionaryMatch(d=diseases, ignore_case=True,
                                        longest_match_only=longest_match_only)
        dict_abbrvs = DictionaryMatch(d=abbrvs, ignore_case=False,
                                      longest_match_only=longest_match_only)

        #
        #  STEM WORDS (diseases + abbrvs + generic disease terms)
        #
        keep = ["disease", "diseases", "syndrome", "syndromes", "disorder",
                "disorders", "damage", "infection", "bleeding"]
        stems = dict.fromkeys(diseases.keys() + abbrvs.keys() + keep)
        disease_stems = DictionaryMatch(d=stems, ignore_case=True,
                                        longest_match_only=longest_match_only)

        #
        # TYPED DISEASES
        #
        # 10406661 type II collagenopathy
        # 8113388 type I protein S deficiency
        # 10807385 stage III cancers
        # 8622978 type IID von Willebrand disease  *** BUG ***
        #
        # 8113388 protein S deficiency type I
        # 8571951 Atelosteogenesis type II
        #
        type_names = ['type', 'class', 'factor']
        type_nums = ['i', 'ii', 'iii', 'vi', 'v', 'vi', '1a', 'iid', 'a', 'b', 'c', 'd']
        type_nums += map(unicode, range(1, 10))

        types = Concat(DictionaryMatch(d=type_names),
                       DictionaryMatch(d=type_nums))

        disease_types_left = Concat(types, disease_stems)
        disease_types_right = Concat(disease_stems, types)

        #
        # DISEASE WITH BODY PART
        #
        body_part = UmlsNoiseAwareDict(positive=["Body Part, Organ, or Organ Component", "Body Location or Region"],
                                       name="*", ignore_case=True).dictionary()

        body_part = {t.lower(): 1 for t in body_part if len(t) > 1 and not t.isdigit()}

        specical_body_part = ['back']

        functional_concept = UmlsNoiseAwareDict(positive=["Functional Concept"],
                                                name="*", ignore_case=True).dictionary()
        functional_concept = {t.lower(): 1 for t in body_part if len(t) > 1 and
                              t.lower() not in stopwords and not t.isdigit()}

        disease_pattern = ['block', 'cancer', 'cancers', 'carcinoma', 'carcinomas', 'damage',
                           'disease', 'diseases', 'disorder', 'disorders', 'dysfunction',
                           'dysfunctions', 'failure', 'failures', 'impairment', 'impairments',
                           'infection', 'injury', 'lesion', 'lesions', 'occlusion', 'occlusions',
                           'pain', 'syndrome', 'syndromes', 'thrombosis', 'toxicity']

        timestamp = ["end-stage", "acute", "chronic", "congestive"]

        conjunction = ["and", "or", "and/or"]

        bf = body_part
        bf.update(functional_concept)

        stemmer = 'porter'
        body_disease = Concat(Concat(DictionaryMatch(d=bf, longest_match_only=longest_match_only, stemmer=stemmer),
                                     DictionaryMatch(d=conjunction, longest_match_only=longest_match_only)),
                              Concat(DictionaryMatch(d=timestamp, longest_match_only=longest_match_only),
                                     Concat(
                                         DictionaryMatch(d=bf, longest_match_only=longest_match_only, stemmer=stemmer),
                                         DictionaryMatch(d=disease_pattern, longest_match_only=longest_match_only,
                                                         stemmer=stemmer)), left_required=False), left_required=False)

        self.matcher = Union(
            disease_types_left, disease_types_right,
            dict_diseases, dict_abbrvs,
            body_disease,
            longest_match_only=longest_match_only)

        print "CdrDiseaseDictGenerator initialized..."