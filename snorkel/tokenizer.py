'''
Apply tokenization to sentence object
'''
import sys
import re
import numpy as np
from collections import defaultdict
from .parser import Sentence

# :
def tokenize(s, split_chars):
    #s = s.replace(u"\xa0", u"_")  # HACK
    s = s.replace(u"\xa0", u" ")  # HACK
    tokens = s.split()
    rgx = r'([{}]+)+'.format("".join(split_chars))
    t_tokens = []
    for t in tokens:
        t = t.replace("'s", " 's")
        t = t.replace("s'", "s '")
        # corenlp replacement tokens
        t = t.replace("``", '"')
        t = t.replace("''", '"')
        t = t.replace("`", "'")
        if not re.search("^[-]*([0-9]+/[0-9]+|[0-9]+[.]*[0-9]*)$",t):
            t = re.sub(rgx, r' \1 ', t)
        t_tokens += [t]
    seq = re.sub("\s{2,}"," "," ".join(t_tokens))
    return seq.split()


    # rgx = r'([{}]+)+'.format("".join(split_chars))
    # seq = re.sub(rgx, r' \1 ', s)
    # seq = seq.replace("'s", " 's")
    # seq = seq.replace("s'", "s '")
    # # corenlp replacement tokens
    # seq = seq.replace("``", '"')
    # seq = seq.replace("''", '"')
    # seq = seq.replace("`", "'")
    # return seq.split()


def retokenize_sentence(sentence, split_chars=["/", "-"]):
    '''Force new tokenization and create new Sentence object'''
    pos_tag_map = {"/": ":", "-": ":"}
    words = [" ".join(tokenize(w, split_chars=split_chars)) for w in sentence["words"]]
    lemmas = [" ".join(tokenize(w, split_chars=split_chars)) for w in sentence["lemmas"]]

    parts = defaultdict(list)
    for i in range(len(words)):
        tokens = words[i].split()
        if len(tokens) != 1:
            # remap offsets
            char_start = sentence["char_offsets"][i]
            char_end = char_start + len(sentence["words"][i]) + 1
            char_idxs = range(char_start, char_end)
            for t in tokens:
                parts["char_offsets"] += [char_idxs[0]]
                char_idxs = char_idxs[len(t):]

                # fix syntactic tags
            pos_tag = sentence["poses"][i]
            parts["poses"] += [pos_tag if t not in pos_tag_map else pos_tag_map[t] for t in tokens]
            parts["dep_labels"] += [sentence["dep_labels"][i]] * len(tokens)
            parts["dep_parents"] += [sentence["dep_parents"][i]] * len(tokens)
            parts["lemmas"] += lemmas[i].split()
            parts["words"] += tokens
        else:
            parts["char_offsets"] += [sentence["char_offsets"][i]]
            parts["words"] += tokens
            parts["lemmas"] += lemmas[i].split()
            parts["poses"] += [sentence["poses"][i]]
            parts["dep_labels"] += [sentence["dep_labels"][i]]
            parts["dep_parents"] += [sentence["dep_parents"][i]]

    # sanity check for character offsets
    for (i, word) in zip(parts["char_offsets"], parts["words"]):
        offset = sentence["char_offsets"][0]
        i -= offset
        j = i + len(word)
        if word != sentence["text"][i:j]:
            print>> sys.stderr, "Warning -- char offset error"
            print>> sys.stderr, word, "|", sentence["text"][i:j]
            print>> sys.stderr, " ".join(parts["words"])

    # santity check for token lengths
    t_sentence = parts.values()
    lengths = map(len, t_sentence)
    if lengths.count(max(lengths)) != len(lengths):
        print>> sys.stderr, "Warning -- sentence conversion error"

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
    for idx in range(len(sentence.char_offsets) - 1):
        i, j = sentence.char_offsets[idx], sentence.char_offsets[idx + 1]
        #if max(candidate.char_start, i) <= min(candidate.char_end, j):
        #    idxs += [idx]
        if set(range(i,j)).intersection(range(candidate.char_start,candidate.char_end+1)):
            idxs += [idx]

    i,j = sentence.char_offsets[-1],len(sentence.text) + sentence.char_offsets[0]
    if set(range(i, j)).intersection(range(candidate.char_start, candidate.char_end + 1)):
        idxs += [len(sentence.char_offsets) - 1]

    return idxs

def tag_sentence(sentence, candidates, tag_fmt="IOB2", split_chars=["/", "-"]):
    sent = retokenize_sentence(sentence) if split_chars else sentence
    ner_tags = np.array(['O'] * len(sent.words))

    c_idx = {}
    for c in candidates:
        idxs = char2idx(sent, c)  # convert to new sentence idxs
        if len(idxs) == 0:
            print>>sys.stderr,"ERROR -- no idxs found", c

        if 'I' in list(ner_tags[idxs]) or 'B' in list(ner_tags[idxs]):
            print>> sys.stderr, "WARNING Double Samples"
            print>> sys.stderr, c
            print>> sys.stderr, zip(sent.words,ner_tags),"\n"
            continue

        if tag_fmt == "IOB2":
            tag_seq = ['B'] + ['I'] * (len(idxs) - 1)
        else:
            tag_seq = ['B'] + ['I'] * (len(idxs) - 1)
            tag_seq = ["S"] if len(idxs) == 1 else tag_seq[0:-1] + ["E"]

        ner_tags[idxs] = tag_seq
        for i in range(min(idxs), max(idxs) + 1):
            c_idx[i] = c

    return (sent, ner_tags)


