from collections import defaultdict
from tokenizers import SpacyTokenizer,RegexTokenizer
from ..models import construct_stable_id
from ..parsers import Parser, ParserConnection


class RuleBasedParser(Parser):
    '''
    Simple, rule-based parser that requires objects  for
     1) detecting sentence boundaries
     2) tokenizing
    '''
    def __init__(self, tokenizer=None, sent_boundary=None):

        super(RuleBasedParser, self).__init__(name="rules")
        self.tokenizer = tokenizer if tokenizer else SpacyTokenizer("en")
        self.sent_boundary = sent_boundary if sent_boundary else RegexTokenizer("[\n\r]+")

    def to_unicode(self, text):

        text = text.encode('utf-8', 'error')
        text = text.decode('string_escape', errors='ignore')
        text = text.decode('utf-8')
        return text

    def connect(self):
        return ParserConnection(self)

    def parse(self, document, text):
        '''
        Transform spaCy output to match CoreNLP's default format
        :param document:
        :param text:
        :return:
        '''
        text = self.to_unicode(text)

        offset, position = 0, 0
        sentences = self.sent_boundary.apply(text)

        for sent,sent_offset in sentences:
            parts = defaultdict(list)
            tokens = self.tokenizer.apply(sent)
            if not tokens:
                continue

            parts['words'], parts['char_offsets'] = zip(*tokens)
            parts['abs_char_offsets'] = [idx + offset for idx in parts['char_offsets']]
            parts['lemmas'] = []
            parts['pos_tags'] = []
            parts['ner_tags'] = []
            parts['dep_parents'] = []
            parts['dep_labels'] = []
            parts['position'] = position

            position += 1
            offset += len(sent)

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = sent

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = parts['abs_char_offsets'][0]
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            if document:
                parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)

            yield parts