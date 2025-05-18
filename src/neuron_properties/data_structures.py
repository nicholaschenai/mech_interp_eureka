"""
Data structures for neuron properties.
"""
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class PhaseType(Enum):
    """Types of phases a neuron can be selective to."""
    APPROACHING = "approaching"
    OPENING = "opening"
    DECELERATION = "deceleration"


class SelectivityMode(Enum):
    """Modes of phase selectivity calculation."""
    OFFSET = "offset"
    ABSOLUTE = "absolute"


@dataclass
class Correlation:
    """Correlation between a neuron and an observation."""
    feature_name: str
    correlation: float
    
    def __post_init__(self):
        """Validate correlation value."""
        if not -1 <= self.correlation <= 1:
            raise ValueError(f"Correlation must be between -1 and 1, got {self.correlation}")


@dataclass
class PhaseSelectivity:
    phase: PhaseType
    selectivity: float
    mode: SelectivityMode
    
    def __post_init__(self):
        """Validate selectivity value."""
        if self.selectivity < 0:
            raise ValueError(f"Selectivity must be non-negative, got {self.selectivity}")


@dataclass
class NodeProperties:
    """Properties of a single neuron node."""
    correlations: List[Correlation] = None
    phase_selectivities: List[PhaseSelectivity] = None
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.correlations is None:
            self.correlations = []
        if self.phase_selectivities is None:
            self.phase_selectivities = []
    
    def add_correlation(self, correlation: Correlation):
        self.correlations.append(correlation)
    
    def add_phase_selectivity(self, phase: PhaseSelectivity):
        self.phase_selectivities.append(phase)
    
    def get_strongest_correlation(self) -> Optional[Correlation]:
        if not self.correlations:
            return None
        return max(self.correlations, key=lambda x: abs(x.correlation))
    
    def get_strongest_phase(self) -> Optional[PhaseSelectivity]:
        """Get the strongest phase selectivity."""
        if not self.phase_selectivities:
            return None
        return max(self.phase_selectivities, key=lambda x: x.selectivity)
    
    def has_properties(self) -> bool:
        return bool(self.correlations or self.phase_selectivities)
