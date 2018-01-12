import os

from snorkel.contrib.babble.pipelines import SnorkelPipeline

from tutorials.qalf.qalf_converter import QalfConverter

class QalfPipeline(SnorkelPipeline):

    def collect(self):
        print("*QalfPipeline objects load QA result matrices in the label() method.")
    
    def label(self):
        
        raise NotImplementedError

        # Legacy code: to be removed
        qc = QalfConverter(self.session, self.candidate_class)
        matrix_path = (os.environ['SNORKELHOME'] + 
            '/tutorials/qalf/spouse/data/qalf_matrix_hp.tsv')
        stats_path = (os.environ['SNORKELHOME'] + 
            '/tutorials/qalf/spouse/data/qalf_stats_hp.tsv')
        L_train, L_dev, L_test = qc.convert(matrix_path, stats_path)

        self.L_train = L_train
        self.L_dev = L_dev
        self.L_test = L_test