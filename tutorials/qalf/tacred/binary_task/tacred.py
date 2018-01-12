"""Utilities for TACRED."""
import argparse
import collections
import os
import sys

OPTS = None

MAIN_DIR = 'data/tacred'
CONLL_DIR = os.path.join(MAIN_DIR, 'conll')
STATS_FILE = os.path.join(MAIN_DIR, 'tacred.stats')

class Example(object):
  def __init__(self, uid, docid, relation, tokens):
    self.uid = uid
    self.doc_id = docid
    self.relation = relation
    self.tokens=tokens

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

Token = collections.namedtuple(
    'Token', ['index', 'token', 'subj', 'subj_type', 'obj', 'obj_type',
              'stanford_pos', 'stanford_ner', 'stanford_deprel',
              'stanford_head'])

def read_relations():
  relations = []
  with open(STATS_FILE) as f:
    for line in f:
      toks = line.strip().split('\t')
      if ':' in toks[0]:
        relations.append(toks[0])
  return relations

def _read_token(line):
  toks = line.split('\t')
  toks[0] = int(toks[0])  # index
  toks[-1] = int(toks[-1])  # stanford_head
  return Token(*toks)

def read_data(split):
  filename = os.path.join(CONLL_DIR, '%s.conll' % split)
  dataset = []
  uid = doc_id = rel = cur_tokens = None
  with open(filename) as f:
    for i, line in enumerate(f):
      if i == 0 and line.startswith('# index'): continue
      line = line.strip()
      if line.startswith('#'):
        uid, doc_id, rel = [x.split('=')[1] for x in line.split(' ')[1:]]
        cur_tokens = []
      elif line:
        tok = _read_token(line)
        cur_tokens.append(tok)
      else:
        dataset.append(Example(uid, doc_id, rel, cur_tokens))
  return dataset

def write_data(filename, dataset):
  with open(filename, 'w') as f:
    for ex in dataset:
      print >> f, '# id=%s doc_id=%s reln=%s' % (ex.uid, ex.doc_id, ex.relation)
      for t in ex.tokens:
        print >> f,'\t'.join(str(x) for x in t)
      print >> f
