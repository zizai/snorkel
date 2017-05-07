"""
Subpackage for learning the structures of models.
"""
from .gen_learning import DependencySelector
from .coral_learning import CoralDependencySelector
from .synthetic import generate_model, generate_label_matrix
from .utils import *
