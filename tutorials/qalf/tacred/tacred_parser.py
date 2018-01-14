import os

import numpy as np

from snorkel.models import Context, Document, Sentence, Candidate
from snorkel.models import construct_stable_id
from snorkel.udf import UDF, UDFRunner

from tutorials.qalf.tacred import TacredReader

PTB = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{', '-RSB-': ']', '-LSB-': '['}

class TacredParser(UDFRunner):

    def __init__(self):
        super(TacredParser, self).__init__(TacredParserUDF)

    def apply(self, xs, split=0, **kwargs):
        super(TacredParser, self).apply(xs, split=split, **kwargs)

    def clear(self, session, **kwargs):
        session.query(Context).delete()
        # We cannot cascade up from child contexts to parent Candidates,
        # so we delete all Candidates too
        session.query(Candidate).delete()
        # session.query(Candidate).filter(Candidate.split == split).delete()


class TacredParserUDF(UDF):

    def __init__(self, **kwargs):
        super(TacredParserUDF, self).__init__(**kwargs)

    def apply(self, example, split=0, **kwargs):
        """Convert each TacredExample into one Doc -> Sentence -> Candidate
        
        Args:
            xs: Iterator of TacredExamples
        """
        doc = self._create_doc(example)
        yield doc
        sentence = self._create_sentence(example, doc)
        yield sentence

    def _create_doc(self, example):
        # Create one Document per TacredExample

        stable_id = "{}::document:0:0".format(example.uid)
        return Document(name=example.uid, 
                        stable_id=stable_id,
                        meta={'doc_id': example.doc_id})

    def _create_sentence(self, example, doc):
        # Create one Sentence per Document

        (indices, tokens, subjs, subj_types, objs, obj_types, pos_tags, 
            ner_tags, _, dep_parents) = zip(*(example.tokens))
        
        parts = {
            'words': [PTB.get(t, t) for t in tokens],
            'pos_tags': pos_tags,
            'ner_tags': ner_tags,
            'dep_parents': dep_parents,
        }
        word_lengths_with_space = map(lambda x: len(x) + 1, tokens)
        char_offsets = list(np.cumsum([0] + word_lengths_with_space[:-1]))
        parts['char_offsets'] = char_offsets
        parts['abs_char_offsets'] = char_offsets

        parts['entity_cids'] = ['O' for _ in parts['words']]
        entity_types = []
        for i, (subj, obj) in enumerate(zip(subjs, objs)):
            if subj == 'SUBJECT':
                entity_types.append('Subject')
            elif obj == 'OBJECT':
                entity_types.append('Object')
            else:
                entity_types.append('O')
        parts['entity_types'] = entity_types

        text = ' '.join(tokens)
        parts['document'] = doc
        parts['position'] = 0  # Each Sentence is the first in its Document
        parts['text'] = text
        parts['stable_id'] = construct_stable_id(doc, 'sentence', 0, len(text))
        
        return Sentence(**parts)