"""
Base class for consciousness theories.
Each theory computes a score from an OrganismState snapshot.
Theories are observers: they read, never modify.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from organism.organism_state import OrganismState


@dataclass
class TheoryScore:
    """Result of a theory computation on one tick."""
    theory: str
    value: float  # 0..1 composite score
    components: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ConsciousnessTheory(ABC):
    """Abstract base for all consciousness theories."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'MDM', 'GWT')."""
        ...

    @abstractmethod
    def compute(self, state: OrganismState) -> TheoryScore:
        """Compute theory score from organism state snapshot."""
        ...
