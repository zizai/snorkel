import csv

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from snorkel.annotations import csr_LabelMatrix, LabelAnnotator

NUM_SPLITS = 3

class QalfConverter(object):
    """Converts a matrix_*.tsv qalf file into csrAnnotationMatrix's."""
    def __init__(self, session, candidate_class):
        self.session = session
        self.candidate_class = candidate_class
    
    def convert(self, matrix_tsv_path, stats_tsv_path):
        candidate_map = {}
        split_sizes = [0] * NUM_SPLITS
        for split in [0, 1, 2]:
            candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).order_by(
                    self.candidate_class.id).all()
            split_sizes[split] = len(candidates)
            for c in candidates:
                candidate_map[c.get_stable_id()] = (c.id, split)

        lf_names = self._extract_lf_names(stats_tsv_path)
        label_matrices = self._tsv_to_matrix(matrix_tsv_path, lf_names, candidate_map)

        for i, label_matrix in enumerate(label_matrices):
            assert(label_matrix.shape[0] == split_sizes[i])
            
        return label_matrices

    def _extract_lf_names(self, stats_tsv_path):
        """
        Args:
            matrix_tsv_path: path to tsv where first column is QA question.
        Returns:
            list of strings containing the questions with spaces replaced by
                underscores.
        """
        lf_names = []
        with open(stats_tsv_path, 'rb') as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:    
                question = row[0]
                lf_names.append('_'.join(question.split()))
        return lf_names

    def _tsv_to_matrix(self, matrix_tsv_path, lf_names, candidate_map):
        """
        Args:
            matrix_tsv_path: path to tsv where first column is candidate_ids
                and all remaining columns contains labels from qa queries.
            lf_names: a list of strings containing the names of the lfs
            candidate_map: dict mapping candidate_ids to their split
        Returns:
            L_train, L_dev, L_test: a csrAnnotationMatrix for each split
        """
        rows = [[], [], []]
        cols = [[], [], []]
        data = [[], [], []]
        row_ids = [[], [], []]
        col_ids = [[], [], []]
        candidate_count = [0] * 3

        misses = 0
        with open(matrix_tsv_path, 'rb') as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                candidate_id = row[0]

                orm_id, split = candidate_map[candidate_id]
                i = candidate_count[split]
                candidate_count[split] += 1

                row_ids[split].append(orm_id)
                
                labels = row[1:]
                if not col_ids[split]:
                    # Just use indices as column ids
                    col_ids[split] = range(len(labels))

                for j, label in enumerate(labels):
                    label = int(label)
                    if label:
                        rows[split].append(i)
                        cols[split].append(j)
                        data[split].append(label)

        label_matrices = [None] * NUM_SPLITS
        for split in [0, 1, 2]:
            coo = coo_matrix((data[split], (rows[split], cols[split])), 
                shape=(len(row_ids[split]), len(col_ids[split])))
            label_matrices[split] = self.coo_to_labelmatrix(coo, row_ids[split], lf_names, split)
        
        return label_matrices

    def coo_to_labelmatrix(self, coo, row_ids, lf_names, split):
        candidate_index = {candidate_id: i for i, candidate_id in enumerate(row_ids)}
        
        def label_generator(c):
            candidate_idx = candidate_index[c.id]
            row = np.ravel(coo.getrow(candidate_idx).todense())
            for i, label in enumerate(row):
                yield lf_names[i], int(label)

        labeler = LabelAnnotator(label_generator=label_generator)
        
        if split == 0:
            L = labeler.apply(split=split, parallelism=1)
        else:
            L = labeler.apply_existing(split=split, parallelism=1)
        return L