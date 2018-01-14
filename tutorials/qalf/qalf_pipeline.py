import os

from snorkel.contrib.babble.pipelines import SnorkelPipeline

from tutorials.qalf.qalf_converter import QalfConverter

class QalfPipeline(SnorkelPipeline):

    def collect(self):
        print("*QalfPipeline objects load QA result matrices in the label() method.")
    
    def label(self, mat_path, split):
        """Converts a .mat qalf matrix into a csr_LabelMatrix for a single split."""
        qc = QalfConverter(self.session, self.candidate_class)
        L = qc.convert(mat_path, split)
        return L