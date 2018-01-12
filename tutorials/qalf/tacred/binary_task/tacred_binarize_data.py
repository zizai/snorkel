"""Extract binary classification problems from TACRED data."""
import argparse
import random
import os
import shutil
import sys

import tacred

OPTS = None

OUT_DIR = 'data/tacred_binary'
DATA_DIR = os.path.join(OUT_DIR, 'data')
STATS_FILE = os.path.join(OUT_DIR, 'stats.txt')
TYPES_FILE = os.path.join(OUT_DIR, 'types.txt')

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--neg-ratio', type=float, default=0.0,
                      help='Proportion of negatives to positives (default=keep all)')
  parser.add_argument('--rng-seed', type=int, default=0)
  return parser.parse_args()

def main():
  random.seed(OPTS.rng_seed)
  if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
  os.makedirs(DATA_DIR)  # Also makes OUT_DIR
  relations = tacred.read_relations()
  relation_to_types = {}
  stats = []
  for split in ('train', 'dev', 'test'):
    data = tacred.read_data(split)
    for rel in relations:
      pos_data = [x for x in data if x.relation == rel]
      if rel in relation_to_types:
        subj_type, obj_type = relation_to_types[rel]
      else:
        subj_type = pos_data[0].subject_type()
        obj_type = pos_data[1].object_type()
        relation_to_types[rel] = (subj_type, obj_type)
      neg_data = [x for x in data 
                  if (x.relation != rel and x.subject_type() == subj_type
                      and x.object_type() == obj_type)]
      if OPTS.neg_ratio and len(neg_data) > OPTS.neg_ratio * len(pos_data):
        num_neg = int(OPTS.neg_ratio * len(pos_data))
        neg_data = random.sample(neg_data, num_neg)
      out_file = os.path.join(DATA_DIR, '%s.%s.conll' % (
          rel.replace(':', '_').replace('/', '_'), split))
      stats.append('%s %s %d %d' % (rel, split, len(pos_data), len(neg_data)))
      combined_data = pos_data + neg_data
      random.shuffle(combined_data)
      tacred.write_data(out_file, combined_data)
  with open(STATS_FILE, 'w') as f:
    print >> f, 'relation split num_pos num_neg'
    for line in stats:
      print >> f, line
  with open(TYPES_FILE, 'w') as f:
    print >> f, 'relation subj_type obj_type'
    for rel in relations:
      subj_type, obj_type = relation_to_types[rel]
      print >> f, '%s %s %s' % (rel, subj_type, obj_type)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

