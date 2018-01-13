import os

from snorkel.models import Document, Sentence

from tutorials.babble.spouse import SpousePipeline
from tutorials.qalf import QalfPipeline
from tutorials.qalf.qalf_converter_legacy import LegacyQalfConverter
from tutorials.qalf.tacred import TacredReader, TacredExtractor

TACRED_DATA = os.path.join(os.environ['SNORKELHOME'], 
    'tutorials/qalf/tacred/binary_task/data')


class TacredQalfPipeline(QalfPipeline):
    
    def parse(self, clear=True):
        # TacredExtractor parses documents, extracts candidates, and loads gold.
        relation = self.config['relation']
        extractor = TacredExtractor(relation)
        for split in self.config['splits']:
            split_name = ['train', 'dev', 'test'][split]
            filename = "{}.{}.conll".format(relation, split_name)
            filepath = os.path.join(TACRED_DATA, filename)            
            reader = TacredReader(filepath)
            extractor.apply(reader,
                            split=split, 
                            parallelism=self.config['parallelism'], 
                            clear=clear)
            # Only clear before the first extraction. 
            # Without this, each split overwrites the previous ones for this run.
            if clear:
                clear = False

    def extract(self):
        # Use PretaggedCandidateExtractor
        print("Candidates were extracted in TacredQalfPipeline.parse() method.")

    def load_gold(self):
        # Loop through candidates, getting gold from sentence's metadata
        print("Gold labels were loaded in TacredQalfPipeline.parse() method.")