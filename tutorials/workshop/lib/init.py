"""
Configure database connection for all workshop notebooks
"""
import os

USE_SQLITE = False

if not USE_SQLITE:
    os.environ['SNORKELDB'] = "postgresql://ubuntu:snorkel@localhost/spouse"

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.models import Candidate, Sentence, Span, Document

session = SnorkelSession()

