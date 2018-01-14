import os

from snorkel.candidates import PretaggedCandidateExtractor
from snorkel.models import Document, Sentence

from tutorials.babble.spouse import SpousePipeline
from tutorials.qalf import QalfPipeline
from tutorials.qalf.qalf_converter_legacy import LegacyQalfConverter
from tutorials.qalf.tacred import TacredReader, TacredParser

TACRED_DATA = os.path.join(os.environ['SNORKELHOME'], 
    'tutorials/qalf/tacred/binary_task/data')


class TacredQalfPipeline(QalfPipeline):
    
    def __init__(self, *args, **kwargs):
        super(TacredQalfPipeline, self).__init__(*args, **kwargs)
        self.sentence_splits = {}
        self.sentence_gold = {}

    def parse(self, clear=True):
        """Parse files into Documents with one Sentence per Document."""
        parser = TacredParser()
        
        for split in self.config['splits']:
            split_name = ['train', 'dev', 'test'][split]
            filename = "{}.{}.conll".format(self.config['relation'], split_name)
            filepath = os.path.join(TACRED_DATA, filename)            
            reader = TacredReader(filepath)
            
            # WARNING: This line brings the entire split into memory
            examples = [e for e in reader]
            
            parser.apply(examples,
                         split=split, 
                         parallelism=self.config['parallelism'], 
                         clear=clear)
                         
            for e in examples:
                # Store sentence split assignments
                self.sentence_splits[e.uid] = split
                # Store sentence-level gold
                self.sentence_gold[e.uid] = (e.relation == 
                    self.config['relation'].replace('_', ':'))

            # Only allow a clear operation before the first parsing. 
            # Without this, each split overwrites the previous ones for this run.
            if clear:
                clear = False

        if self.config['verbose']:
            num_docs = self.session.query(Document).count()
            print("Parsed {} Documents".format(num_docs))

            num_sents = self.session.query(Sentence).count()
            print("Parsed {} Sentences".format(num_sents))

    def extract(self, clear=True):
        """Extract candidates using PretaggedCandidateExtractor."""
        if not self.sentence_splits:
            raise Exception("You must run .parse() before running .extract()")

        candidate_extractor = PretaggedCandidateExtractor(
            self.candidate_class, ['Subject', 'Object'])

        sentences_by_split = {split: [] for split in self.config['splits']}
        sentences = self.session.query(Sentence).all()
        for sent in sentences:
            uid = sent.stable_id.split('::')[0]
            sentences_by_split[self.sentence_splits[uid]].append(sent)

        for split in self.config['splits']:
            candidate_extractor.apply(sentences_by_split[split], 
                                      split=split, parallelism=1, clear=clear)

            # Only allow a clear operation before the first extraction. 
            # Without this, each split overwrites the previous ones for this run.
            if clear:
                clear = False

            if self.config['verbose']:
                num_cands = self.session.query(self.candidate_class).filter(
                    self.candidate_class.split == split).count()
                print("Split {}: Extracted {} Candidates".format(split, num_cands))

    def load_gold(self):
        # Loop through candidates, getting gold from sentence's metadata
       raise NotImplementedError