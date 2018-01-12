#!/usr/bin/env python

"""
Generate json format TACRED data, based on the conll data.
The generated data will be written into the ../json directory.
"""

import os
import json
import argparse
from collections import OrderedDict

HAS_HEADER = True
NUM_FIELD = 10

splits = ['train', 'dev', 'test']

def process_all(conll_dir, json_dir):
    for sp in splits:
        data_file = conll_dir + '/' + sp + '.conll'
        examples = []
        cache = []
        with open(data_file) as infile:
            if HAS_HEADER:
                header = infile.readline().strip()
            while True:
                line = infile.readline()
                if len(line) == 0:
                    break
                line = line.strip()
                if len(line) == 0: # empty after strip
                    if len(cache) > 0:
                        examples.append(cache)
                        cache = []
                else:
                    cache.append(line)
            if len(cache) > 0:
                examples.append(cache)
        # read finish
        print("{} examples read in {} split.".format(len(examples), sp))
    
        json_data = convert_to_json(examples)
    
        with open(json_dir + '/' + sp + '.json', 'w') as outfile:
            json.dump(json_data, outfile)
    
        print("{} data dumped to json file.".format(sp))

def convert_to_json(examples):
    json_examples = []
    for ex in examples:
        json_instance = OrderedDict()
        head = ex[0].split(' ')
        hashcode = head[1][3:]
        docid = head[2][6:]
        rel = head[3][5:]
        conll_array = [x.split('\t') for x in ex[1:]]
        # load
        tokens, objects, subjects, stanford_pos, stanford_ner, stanford_head, stanford_deprel = \
                [], [], [], [], [], [], []
        subj_type, obj_type = '', ''
        for arr in conll_array:
            assert(len(arr) == NUM_FIELD)
            arr = arr[1:] # eat index
            tokens.append(arr[0])
            subjects.append(arr[1])
            objects.append(arr[3])
            if arr[2] != '_':
                subj_type = arr[2]
            if arr[4] != '_':
                obj_type = arr[4]
            stanford_pos.append(arr[5])
            stanford_ner.append(arr[6])
            stanford_deprel.append(arr[7])
            stanford_head.append(arr[8])
        assert(subj_type != '')
        assert(obj_type != '')
        subj_start, subj_end, obj_start, obj_end = -1, -1, -1, -1
        for i, (subj, obj) in enumerate(zip(subjects, objects)):
            if subj == 'SUBJECT':
                if subj_start == -1:
                    subj_start = i
                subj_end = i
            if obj == 'OBJECT':
                if obj_start == -1:
                    obj_start = i
                obj_end = i
        assert(subj_start >= 0)
        assert(subj_end >= 0)
        assert(obj_start >= 0)
        assert(obj_end >= 0)
        # set
        json_instance['id'] = hashcode
        json_instance['docid'] = docid
        json_instance['relation'] = rel
        json_instance['token'] = tokens
        json_instance['subj_start'] = subj_start
        json_instance['subj_end'] = subj_end
        json_instance['obj_start'] = obj_start
        json_instance['obj_end'] = obj_end
        json_instance['subj_type'] = subj_type
        json_instance['obj_type'] = obj_type
        json_instance['stanford_pos'] = stanford_pos 
        json_instance['stanford_ner'] = stanford_ner
        json_instance['stanford_head'] = stanford_head
        json_instance['stanford_deprel'] = stanford_deprel
        json_examples.append(json_instance)
    return json_examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate json data based on conll format data.')
    parser.add_argument('conll_dir', type=str, help='Directory where the conll format data exists')
    parser.add_argument('json_dir', type=str, help='Directory where the json format data should be written.')
    args = parser.parse_args()

    if not os.path.exists(args.json_dir):
        os.mkdir(args.json_dir)
    process_all(args.conll_dir, args.json_dir)
