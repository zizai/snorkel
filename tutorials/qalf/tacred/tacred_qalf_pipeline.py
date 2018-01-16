import os

from snorkel.candidates import PretaggedCandidateExtractor
from snorkel.db_helpers import reload_annotator_labels
from snorkel.models import Document, Sentence
from snorkel.models import StableLabel
from snorkel.utils import ProgressBar

from tutorials.babble.spouse import SpousePipeline
from tutorials.qalf import QalfPipeline
from tutorials.qalf.qalf_converter_legacy import LegacyQalfConverter
from tutorials.qalf.tacred import TacredReader, TacredParser

TACRED_BINARY = os.path.join(os.environ['SNORKELHOME'], 
    'tutorials/qalf/tacred/binary_task/')
TACRED_DATA = os.path.join(TACRED_BINARY, 'data')
TACRED_MATRICES = os.path.join(TACRED_BINARY, 'matrices')

SPLIT_NAMES = ['train', 'dev', 'test']

class TacredQalfPipeline(QalfPipeline):
    
    def __init__(self, *args, **kwargs):
        super(TacredQalfPipeline, self).__init__(*args, **kwargs)
        self.sentence_splits = {}
        self.sentence_gold = {}

    def parse(self, clear=True):
        """Parse files into Documents with one Sentence per Document."""
        parser = TacredParser()
        
        for split in self.config['splits']:
            split_name = SPLIT_NAMES[split]
            filename = "{}.{}.conll".format(self.config['relation'], split_name)
            filepath = os.path.join(TACRED_DATA, filename)            
            reader = TacredReader(filepath)
            
            # WARNING: This line brings the entire split into memory
            # This is done so that we can fill sentence_splits and sentence_gold
            # without having to read the data from file more than once.
            examples = [e for e in reader]

            parser.apply(examples,
                         split=split, 
                         parallelism=self.config['parallelism'], 
                         clear=clear)
                         
            for e in examples:
                # Store sentence split assignments
                self.sentence_splits[e.uid] = split
                # Store sentence-level gold, adjusted from {0, 1} -> {-1, 1}
                self.sentence_gold[e.uid] = (e.relation == 
                    self.config['relation'].replace('_', ':')) * 2 - 1

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
            self.candidate_class, ['Subject', 'Object'], 
            entity_sep=',',
            nested_relations=True)

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

    def load_gold(self, annotator='gold'):
        # Loop through candidates, getting gold from sentence's metadata
        for split in self.config['splits']:
            candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).all()

            pb = ProgressBar(len(candidates))
            for i, c in enumerate(candidates):
                pb.bar(i)
                # Get stable ids and check to see if label already exits
                context_stable_ids = '~~'.join(x.get_stable_id() for x in c)
                query = self.session.query(StableLabel).filter(
                    StableLabel.context_stable_ids == context_stable_ids)
                query = query.filter(StableLabel.annotator_name == annotator)
                # If does not already exist, add label
                uid = c.get_parent().stable_id.split('::')[0]
                label = self.sentence_gold[uid]
                if query.count() == 0:
                    self.session.add(StableLabel(
                        context_stable_ids=context_stable_ids,
                        annotator_name=annotator,
                        value=label))
            pb.close()

            self.session.commit()

            # Reload annotator labels
            reload_annotator_labels(self.session, self.candidate_class, annotator,
                                    split=split, filter_label_split=False)

    def label(self):
        for split in self.config['splits']:
            split_name = SPLIT_NAMES[split]
            filename = "{}.{}.mat".format(self.config['relation'], split_name)
            filepath = os.path.join(TACRED_MATRICES, filename)    
            L = super(TacredQalfPipeline, self).label(filepath, split)

            # TEMP (for debugging convenience)
            if split == 0:
                self.L_train = L
            elif split == 1:
                self.L_dev = L
            elif split == 2:
                self.L_test = L
            else:
                raise ValueError