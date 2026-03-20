# CRISTAL - Consciousness Theories
# All theories implement ConsciousnessTheory ABC from base.py

from .base import ConsciousnessTheory, TheoryScore
from .mdm import MDMTheory
from .gwt import GWTTheory
from .hot import HOTTheory
from .fep import FEPTheory
from .iit import IITTheory
from .dyn import DYNTheory
from .rpt import RPTTheory
from .hybrid import HybridTheory

# Backward compatibility aliases
from .iit import IITImplementation
from .gwt import GlobalWorkspace as GWTImplementation
from .hot import HigherOrderThought as HOTImplementation
from .fep import FreeEnergyPrinciple as FEPImplementation
from .dyn import DynamicalSynchrony as DYNImplementation
from .rpt import RecurrentProcessing as RPTImplementation
from .hybrid import AdaptiveHybrid as HybridImplementation

# All theory classes for programmatic use
ALL_THEORIES = [
    MDMTheory,
    GWTTheory,
    HOTTheory,
    FEPTheory,
    IITTheory,
    DYNTheory,
    RPTTheory,
    HybridTheory,
]

__all__ = [
    'ConsciousnessTheory', 'TheoryScore',
    'MDMTheory', 'GWTTheory', 'HOTTheory', 'FEPTheory',
    'IITTheory', 'DYNTheory', 'RPTTheory', 'HybridTheory',
    'ALL_THEORIES',
    # Backward compat
    'IITImplementation', 'GWTImplementation', 'HOTImplementation',
    'FEPImplementation', 'DYNImplementation', 'RPTImplementation',
    'HybridImplementation',
]
