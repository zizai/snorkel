import os
os.environ['SNORKELDB'] = 'postgres://localhost:5438/cdr-scale-up'
from snorkel import SnorkelSession
session = SnorkelSession()

from six.moves.cPickle import load
from snorkel.models import SequenceTag, Document
from snorkel.parser import XMLMultiDocPreprocessor

doc_preprocessor = XMLMultiDocPreprocessor(
    path="",
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()'
)

# Create canonical CDR folds
folds = {}
with open('data/doc_ids.pkl', 'rb') as f:
    train_ids, dev_ids, test_ids = load(f)
folds[0] = set(train_ids)
folds[1] = set(dev_ids)
folds[2] = set(test_ids)

print len(folds[0])
print len(folds[1])
print len(folds[2])

import itertools
from custom_cand_generator import SequenceTagCandidateExtractor

# bin sentences by CDR fold
sentences = {}
for split in folds:
    documents = session.query(Document).filter(Document.name.in_(folds[split])).all() 
    sentences[split] = list(itertools.chain.from_iterable([doc.sentences for doc in documents]))
    print split, "docs:", len(folds[split]), "sentences:", len(sentences[split]) 


from snorkel.models import Candidate, candidate_subclass

ChemicalDisease = candidate_subclass('ChemicalDisease', ['chemical', 'disease'])

candidate_extractor1 = SequenceTagCandidateExtractor(
    ChemicalDisease, ['Disease', 'Chemical'], tag_sources=['TaggerOne']
)

for split, sents in sentences.items():
    #print split, "docs:", len(folds[split]), "sentences:", len(sentences[split]) 
    candidate_extractor1.apply(list(set(sents)), clear=True, split=split)
    print "Number of candidates:", session.query(ChemicalDisease).filter(ChemicalDisease.split == split).count()
    print


from snorkel.models import Candidate

train = session.query(Candidate.id).filter(Candidate.split==0).all()
dev = session.query(Candidate.id).filter(Candidate.split==1).all()
test = session.query(Candidate.id).filter(Candidate.split==2).all()

rand = session.query(Candidate.id).filter(Candidate.split==3).all()
query = session.query(Candidate.id).filter(Candidate.split==4).all()

print len(train)
print len(dev)
print len(test)
print len(rand)
print len(query)

