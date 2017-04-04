#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Apply tokenization to sentence object
'''
import sys
import re
import numpy as np
from collections import defaultdict
from .parser import Sentence
import itertools

unicode_map = {u'‘':u"'",
               u"’s":u"'s",
               u'’':u"'",
               u"”":u'"',
               u"“":u'"',
               u"…":u"...",
               u"€":u"$",
               u"'":""}

core_nlp = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}',
            '-LCB-': '{', '-RSB-': ']', '-LSB-': '['}

def tokenize(s, split_chars):

    s = s.replace(u"\xa03", u"_")  # HACK
    tokens = s.split()
    rgx = r'([{}]+)+'.format(u"".join(sorted(split_chars)))
    t_tokens = []
    for t in tokens:

        t = t.replace(u"'s", u" 's")
        t = t.replace(u"s'", u"s '")

        # corenlp replacement tokens
        t = t.replace(u"``", u'"')
        t = t.replace(u"''", u'"')
        t = t.replace(u"`", u"'")
        # for PTB-style tokens embedded in words (how does this happen??)
        # e.g., ";-RRB-"
        for ptb in core_nlp:
            if ptb in t:
                t = t.replace(ptb,core_nlp[ptb])

        if re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',t):
            pass
        elif t == "--":
            continue
        elif not re.search("^[-]*([0-9]+/[0-9]+|[0-9]+[.]*[0-9]*)$",t):
            t = re.sub(rgx, r' \1 ', t)

        t_tokens += [t]

    seq = re.sub(u"\s{2,}",u" ",u" ".join(t_tokens))
    return seq.split()

def retokenize_sentence(sentence, split_chars=[u"/", u"-", u"+"]):
    '''Force new tokenization and create new Sentence object'''
    try:
        sentence = sentence._asdict()
    except:
        pass

    pos_tag_map = {u"/": u":", u"-": u":"}
    words = [u" ".join(tokenize(w, split_chars=split_chars)) for w in sentence["words"]]
    lemmas = [u" ".join(tokenize(w, split_chars=split_chars)) for w in sentence["lemmas"]]

    parts = defaultdict(list)
    for i in range(len(words)):
        tokens = words[i].split()
        if len(tokens) != 1:

            # remap offsets
            char_start = sentence["char_offsets"][i]

            # use length of orginal text span
            # ---------------------
            j = sentence["char_offsets"][i+1] - sentence["char_offsets"][0] if i < len(sentence["char_offsets"])-1 else None
            text_span = sentence["text"][char_start-sentence["char_offsets"][0]:j]
            # ---------------------

            char_end = char_start + len(text_span.strip()) + 1
            char_idxs = range(char_start, char_end)

            for t in tokens:
                start = char_idxs[0]
                # if this offset was originally a space, increment char_start
                while sentence["text"][start-sentence["char_offsets"][0]] == " ":
                    char_idxs = char_idxs[1:]
                    start = char_idxs[0]

                parts["char_offsets"] += [char_idxs[0]]
                char_idxs = char_idxs[len(t):]

            # fix syntactic tags
            pos_tag = sentence["poses"][i]
            parts["poses"] += [pos_tag if t not in pos_tag_map else pos_tag_map[t] for t in tokens]
            parts["dep_labels"] += [sentence["dep_labels"][i]] * len(tokens)
            parts["dep_parents"] += [[sentence["dep_parents"][i]] * len(tokens)]
            parts["lemmas"] += lemmas[i].split()
            parts["words"] += tokens

        else:
            parts["char_offsets"] += [sentence["char_offsets"][i]]
            parts["words"] += tokens
            parts["lemmas"] += lemmas[i].split()
            parts["poses"] += [sentence["poses"][i]]
            parts["dep_labels"] += [sentence["dep_labels"][i]]
            parts["dep_parents"] += [[sentence["dep_parents"][i]]]

    # repair dependency tree
    # ---------------------
    shift = defaultdict(int)
    for t in parts["dep_parents"]:
        if len(t) > 1:
            shift[t[0]] += len(t) - 1

    parts["dep_parents"] = [t for t in itertools.chain.from_iterable(parts["dep_parents"])]

    transform = {}
    for parent in sorted(set(parts["dep_parents"])):
        curr_shift = 0
        for offset in shift:
            if parent >= offset:
                curr_shift += shift[offset]
        transform[parent] = parent + curr_shift

    parts["dep_parents"] = [ transform[t] for t in parts["dep_parents"]]
    # ---------------------

    # sanity check for character offsets
    for (i, word) in zip(parts["char_offsets"], parts["words"]):
        offset = sentence["char_offsets"][0]
        i -= offset
        j = i + len(word)
        # some characters are automatically substituted by CoreNLP, so we won't have an exact string match
        # (some of) these characters are stored in unicode_map
        if word != sentence["text"][i:j] and sentence["text"][i:j] not in unicode_map:
            continue
            print>> sys.stderr, "Warning -- char offset error"
            print>> sys.stderr, word, "|", sentence["text"][i:j], i, j, len(word)
            print>> sys.stderr, [sentence["text"]],"\n"
            print>> sys.stderr, " ".join(parts["words"]), "\n"

    # sanity check for token lengths
    t_sentence = parts.values()
    lengths = map(len, t_sentence)
    if lengths.count(max(lengths)) != len(lengths):
        print>> sys.stderr, "Warning -- sentence conversion error"
        return None

    parts['text'] = sentence["text"]
    parts['sent_id'] = sentence["sent_id"]
    parts['doc_id'] = sentence["doc_id"]
    parts['doc_name'] = sentence["doc_name"]
    parts['xmltree'] = None
    parts['id'] = sentence["id"]

    return Sentence(**parts)


def char2idx(sentence, candidate):
    '''char_offsets converted to word idxs'''
    idxs = []

    try:
        N = len(sentence.char_offsets)
    except:
        N = len(sentence['char_offsets'])

    for idx in range(N - 1):
        try:
            i, j = sentence.char_offsets[idx], sentence.char_offsets[idx + 1]
        except:
            i, j = sentence['char_offsets'][idx], sentence['char_offsets'][idx + 1]

        if set(range(i,j)).intersection(range(candidate.char_start,candidate.char_end+1)):
            idxs += [idx]

    try:
        i,j = sentence.char_offsets[-1],len(sentence.text) + sentence.char_offsets[0]
    except:
        i, j = sentence['char_offsets'][-1], len(sentence['text']) + sentence['char_offsets'][0]

    if set(range(i, j)).intersection(range(candidate.char_start, candidate.char_end + 1)):
        try:
            idxs += [len(sentence.char_offsets) - 1]
        except:
            idxs += [len(sentence['char_offsets']) - 1]

    return idxs




def tag_sentence(sentence, candidates, tag_fmt="IOB2", split_chars=["/", "-"]):

    sent = retokenize_sentence(sentence) if split_chars else sentence
    if sent is None:
        return None,None

    try:
        ner_tags = np.array([u'O'] * len(sent.words))
    except:
        ner_tags = np.array([u'O'] * len(sent['words']))

    c_idx = {}
    for c in candidates:
        idxs = char2idx(sent, c)  # convert to new sentence idxs
        if len(idxs) == 0:
            print>>sys.stderr,"ERROR -- no idxs found", c
            continue

        if 'I' in list(ner_tags[idxs]) or 'B' in list(ner_tags[idxs]):
            print>> sys.stderr, "WARNING Double Samples"
            print>> sys.stderr, c
            # print>> sys.stderr, zip(sent.words,ner_tags),"\n"
            continue

        if tag_fmt == "IOB2":
            tag_seq = [u'B'] + [u'I'] * (len(idxs) - 1)
        else:
            tag_seq = [u'B'] + [u'I'] * (len(idxs) - 1)
            tag_seq = [u"S"] if len(idxs) == 1 else tag_seq[0:-1] + [u"E"]

        ner_tags[idxs] = tag_seq
        for i in range(min(idxs), max(idxs) + 1):
            c_idx[i] = c

    return (sent, ner_tags)


