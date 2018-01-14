import csv

import numpy as np
from scipy.sparse import csr_matrix

from snorkel.annotations import csr_LabelMatrix

from snorkel.contrib.babble import sparse_to_labelmatrix

class QalfConverter(object):
    """Converts a .mat qalf matrix file into a csrAnnotationMatrix."""
    def __init__(self, session, candidate_class):
        self.session = session
        self.candidate_class = candidate_class
    
    def convert(self, mat_path, split):
        """
        Args:
            mat_path: path to qalf .mat file
                The first row is a header
                For all other rows, the first column is candidate_ids
                All remaining columns contain labels from QA queries.
            split: int: the split the mat file corresponds to
        Returns:
            L: a csr_LabelMatrix for this split
        """
        rows = []
        cols = []
        data = []
        row_ids = []
        col_ids = []
        
        candidates = self.session.query(self.candidate_class).filter(
            self.candidate_class.split == split).all()
        candidate_id_map = {c.get_stable_id().split('::')[0]: c.id for c in candidates}
        if len(candidates) != len(candidate_id_map):
            raise Exception("More than one candidate were found in the same sentence.")
        num_candidates = len(candidates)

        with open(mat_path, 'rb') as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')

            header = tsv_reader.next()
            lf_names = header[1:]
            num_lfs = len(lf_names)

            for i, row in enumerate(tsv_reader):
                candidate_id = row[0]
                orm_id = candidate_id_map[candidate_id]
                row_ids.append(orm_id)
                
                for j, label in enumerate(row[1:]):
                    label = int(label)
                    if label:
                        rows.append(i)
                        cols.append(j)
                        data.append(label)

        csr = csr_matrix((data, (rows, cols)), 
            shape=(num_candidates, num_lfs))
        # Build a map from a candidate's orm id to its row in the matrix
        candidate_row_map = {candidate_orm_id: i for i, candidate_orm_id in enumerate(row_ids)}
        L = sparse_to_labelmatrix(csr, candidate_row_map, lf_names, split)
        
        return L
