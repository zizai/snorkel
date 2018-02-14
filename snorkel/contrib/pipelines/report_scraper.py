from collections import defaultdict
import json
import os
import sys

parameters = ["max_train"]

date_str = sys.argv[1] # e.g., '02_13_18'                                                        
exp_name = sys.argv[2] # e.g., 'cdr_finaltradit0'
include_all = True if sys.argv[3].lower().startswith('t)') else False # e.g., True

report_subdir = os.path.join(os.environ['SNORKELHOME'], 'reports', date_str)
reports = []
for root, dirs, files in os.walk(report_subdir):
    if exp_name in root:
        for filename in files:
            if filename.endswith('json'):
                reports.append(root + '/' + filename)

results = defaultdict(list)
for report in reports:
    data = json.load(open(report))
    f1 = data["scores"]["F1 Score"]["Disc"]
    r = data["scores"]["Recall"]["Disc"]
    p = data["scores"]["Precision"]["Disc"]
    settings = []
    for parameter in parameters:
        settings.append(data["config"][parameter])
    if include_all:
        results[tuple(settings)].append((p, r, f1))
    else:
        results[tuple(settings)].append((f1,))

for settings, scores_list in sorted(results.items()):
    for param, val in zip(parameters, settings):
        print("{} = {}".format(param, val))
    for scores in scores_list:
        print('\t'.join(map(str, scores)))
    print("")