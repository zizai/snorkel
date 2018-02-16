import argparse
from collections import defaultdict
import json
import os
import sys

def main(args):
    params = [args['param']]

    report_subdir = os.path.join(os.environ['SNORKELHOME'], 'reports', args['date'])
    reports = []
    for root, dirs, files in os.walk(report_subdir):
        if args['exp'] in root:
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
        for parameter in args['params']:
            settings.append(data["config"][parameter])
        if args['f1']:
            results[tuple(settings)].append((f1,))
        else:
            results[tuple(settings)].append((p, r, f1))

    for settings, scores_list in sorted(results.items()):
        for param, val in zip(args['params'], settings):
            print("{} = {}".format(param, val))
        for scores in scores_list:
            print('\t'.join(map(str, scores)))
        print("")


if __name__ == '__main__':
    # Parse command-line args
    argparser = argparse.ArgumentParser(description="Record scraper.")
    argparser.add_argument('--exp', type=str)
    argparser.add_argument('--date', type=str, default='02_15_18')
    argparser.add_argument('--f1', action='store_true')
    argparser.add_argument('--param', type=str, default='max_train')
    # TODO: allow them to specify multiple param values
    args = vars(argparser.parse_args())
    main(args)


