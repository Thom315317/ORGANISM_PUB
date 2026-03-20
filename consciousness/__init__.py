#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRISTAL v13 - Consciousness Module
==================================

Ce module implémente 7 théories de la orchestrator:
- IIT (Integrated Information Theory)
- GWT (Global Workspace Theory)
- HOT (Higher-Order Thought)
- FEP (Free Energy Principle)
- DYN (Dynamical Systems)
- RPT (Recurrent Processing Theory)
- Hybrid (Combinaison adaptative)
"""

# Import des théories
try:
    from .theories.iit import IITImplementation
    from .theories.gwt import GlobalWorkspace
    from .theories.hot import HigherOrderThought
    from .theories.fep import FreeEnergyPrinciple
    from .theories.dyn import DynamicalSynchrony
    from .theories.rpt import RecurrentProcessing
    from .theories.hybrid import AdaptiveHybrid
    
    THEORIES_AVAILABLE = True
except ImportError as e:
    THEORIES_AVAILABLE = False
    print(f"⚠️ Theories import failed: {e}")

# Import des modes phi
try:
    from .phi_modes import PhiModes
except ImportError:
    PhiModes = None

# Import du testbed
try:
    from .testbed import ConsciousnessTestbed
except ImportError:
    ConsciousnessTestbed = None

# Import du wrapper PyPhi
try:
    from .pyphi_wrapper import PyPhiWrapper, compute_exact
    PYPHI_AVAILABLE = True
except ImportError:
    PyPhiWrapper = None
    compute_exact = None
    PYPHI_AVAILABLE = False

__all__ = [
    'IITImplementation',
    'GlobalWorkspace', 
    'HigherOrderThought',
    'FreeEnergyPrinciple',
    'DynamicalSynchrony',
    'RecurrentProcessing',
    'AdaptiveHybrid',
    'PhiModes',
    'ConsciousnessTestbed',
    'PyPhiWrapper',
    'THEORIES_AVAILABLE',
    'PYPHI_AVAILABLE',
]

