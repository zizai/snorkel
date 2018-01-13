import os

from snorkel.models import Context, Document, Sentence, Candidate
from snorkel.models import construct_stable_id
from snorkel.udf import UDF, UDFRunner

from tutorials.qalf.tacred import TacredReader


class TacredExtractor(UDFRunner):

    def __init__(self, relation):
        super(TacredExtractor, self).__init__(TacredExtractorUDF)

    def apply(self, xs, split=0, **kwargs):
        super(TacredExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, **kwargs):
        session.query(Context).delete()
        # We cannot cascade up from child contexts to parent Candidates,
        # so we delete all Candidates too
        session.query(Candidate).delete()
        # session.query(Candidate).filter(Candidate.split == split).delete()


class TacredExtractorUDF(UDF):

    def __init__(self, **kwargs):
        super(TacredExtractorUDF, self).__init__(**kwargs)

    def apply(self, example, split=0, **kwargs):
        """Convert each TacredExample into one Doc -> Sentence -> Candidate
        
        Args:
            xs: Iterator of TacredExamples
        """
        # Create one Document per TacredExample
        stable_id = "{}::document:0:0".format(example.uid)
        doc = Document(
            name=example.uid, stable_id=stable_id,
            meta={'doc_id': example.doc_id}
        )
        yield doc
            # create Sentence
            # create Candidate
            # store Gold
        # Don't forget to commit!
        # print(i)