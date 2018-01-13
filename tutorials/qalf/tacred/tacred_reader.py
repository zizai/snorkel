import collections


Token = collections.namedtuple(
    'Token', ['index', 'token', 'subj', 'subj_type', 'obj', 'obj_type',
              'stanford_pos', 'stanford_ner', 'stanford_deprel',
              'stanford_head'])


class TacredExample(object):
    def __init__(self, uid, docid, relation, tokens):
        self.uid = uid
        self.doc_id = docid
        self.relation = relation
        self.tokens=tokens

    def __repr__(self):
        return "TacredExample(id={}, rel={})".format(self.uid, self.relation)

    def subject_type(self):
        for t in self.tokens:
            if t.subj_type != '_':
                return t.subj_type
        raise ValueError

    def object_type(self):
        for t in self.tokens:
            if t.obj_type != '_':
                return t.obj_type
        raise ValueError


class TacredReader(object):
    """Accepts a path to a .conll file and yields TacredExamples."""
    def __init__(self, filepath):
        self.filepath = filepath

    def generate(self):

        def _read_token(line):
            toks = line.split('\t')
            toks[0] = int(toks[0])  # index
            toks[-1] = int(toks[-1])  # stanford_head
            return Token(*toks)
        
        uid = doc_id = rel = cur_tokens = None
        with open(self.filepath) as f:
            for i, line in enumerate(f):
                line = line.strip().decode('utf-8')
                if i == 0 and line.startswith('# index'): continue
                if line.startswith('#'):
                    uid, doc_id, rel = [x.split('=')[1] for x in line.split(' ')[1:]]
                    cur_tokens = []
                elif line:
                    tok = _read_token(line)
                    cur_tokens.append(tok)
                else:
                    yield TacredExample(uid, doc_id, rel, cur_tokens)
    
    def __iter__(self):
        return self.generate()