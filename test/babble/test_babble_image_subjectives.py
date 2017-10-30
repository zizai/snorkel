import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.contrib.babble import SemanticParser

from test_babble_image import TestBabbleImage
import image_explanations

class TestBabbleImageSubjectives(TestBabbleImage):

    @classmethod
    def setUpClass(cls):
        cls.sp = SemanticParser(mode='image')
        cls.subjectives = True

suite = unittest.TestLoader().loadTestsFromTestCase(TestBabbleImageSubjectives)
unittest.TextTestRunner(verbosity=2).run(suite)
