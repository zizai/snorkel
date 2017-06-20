import re
from .spacy_parser import Spacy


class Tokenizer(object):
    '''
    Interface for rule-based tokenizers
    '''
    def apply(self,s):
        raise NotImplementedError()


class RegexTokenizer(Tokenizer):
    '''
    Regular expression tokenization.
    '''
    def __init__(self, rgx="\s+"):
        super(RegexTokenizer, self).__init__()
        self.rgx = re.compile(rgx)

    def apply(self,s):
        '''
        Apply regex rule for tokenization
        :param s:
        :return:
        '''
        tokens = []
        offset = 0
        # keep track of char offsets
        for t in self.rgx.split(s):
            while t < len(s) and t != s[offset:len(t)]:
                offset += 1
            tokens += [(t,offset)]
            offset += len(t)
        return tokens


class SpacyTokenizer(Tokenizer):
    '''
    Only use spaCy's tokenizer functionality
    '''
    def __init__(self, lang='en'):
        super(SpacyTokenizer, self).__init__()
        self.lang = lang
        self.model = Spacy.load_lang_model(lang)

    def apply(self, s):
        doc = self.model.tokenizer(s)
        return [(t.text, t.idx) for t in doc]
