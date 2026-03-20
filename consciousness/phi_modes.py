"""
Phi Modes - 4 versions de calcul Φ
Toggle entre différentes implémentations.
"""

from typing import Dict, Any
import numpy as np

class PhiCalculator:
    """Calcule Φ selon 4 modes différents."""
    
    MODES = ['simple', 'iit_lite', 'iit_rigorous', 'adaptive']
    
    def __init__(self, mode='simple'):
        self.mode = mode
        self.history = []  # Pour mode adaptive
    
    def compute(self, state: Dict[str, Any]) -> float:
        """Calcule Φ selon mode actif."""
        if self.mode == 'simple':
            return self._phi_simple(state)
        elif self.mode == 'iit_lite':
            return self._phi_iit_lite(state)
        elif self.mode == 'iit_rigorous':
            return self._phi_iit_rigorous(state)
        elif self.mode == 'adaptive':
            return self._phi_adaptive(state)
        else:
            return self._phi_simple(state)
    
    def _phi_simple(self, state: Dict) -> float:
        """Φ heuristique actuel CRISTAL."""
        base = 0.3
        
        # Facteurs
        if state.get('negotiation_cycles', 0) > 0:
            base += 0.2
        if state.get('consensus_reached'):
            base += 0.3
        if state.get('complexity_high'):
            base += 0.2
        
        return min(1.0, base)
    
    def _phi_iit_lite(self, state: Dict) -> float:
        """Approximation IIT variationnelle."""
        # Information mutuelle
        mi = 0.0
        
        active_instances = sum([
            state.get('orchestrator_active', False),
            state.get('creative_sampler_active', False),
            state.get('safety_controller_active', False),
            state.get('context_keeper_active', False)
        ])
        
        if active_instances >= 2:
            mi = active_instances / 4.0 * 0.6
        
        # Intégration
        integration = state.get('negotiation_cycles', 0) / 3.0 * 0.4
        
        # Φ lite
        phi = mi + integration
        return min(1.0, phi)
    
    def _phi_iit_rigorous(self, state: Dict) -> float:
        """IIT rigoureux (approximation car PyPhi lourd)."""
        # Simplification: mesure intégration irréductible
        
        # Total information
        total_info = self._total_information(state)
        
        # Partitioned information
        partition_info = self._min_partition_information(state)
        
        # Φ = intégration irréductible
        phi = max(0.0, total_info - partition_info)
        
        return min(1.0, phi)
    
    def _phi_adaptive(self, state: Dict) -> float:
        """Φ qui apprend quelle formule corrèle mieux."""
        # Calculer tous
        simple = self._phi_simple(state)
        lite = self._phi_iit_lite(state)
        rigorous = self._phi_iit_rigorous(state)
        
        # Si pas assez d'historique, moyenne
        if len(self.history) < 10:
            return (simple + lite + rigorous) / 3.0
        
        # Sinon pondérer selon performances passées
        # (simplifié: moyenne pondérée apprise)
        weights = self._learned_weights()
        
        adaptive_phi = (
            weights['simple'] * simple +
            weights['lite'] * lite +
            weights['rigorous'] * rigorous
        )
        
        return adaptive_phi
    
    def _total_information(self, state: Dict) -> float:
        """Information totale du système."""
        info = 0.0
        
        # Par instance active
        if state.get('orchestrator_active'):
            info += 0.4
        if state.get('creative_sampler_active'):
            info += 0.3
        if state.get('safety_controller_active'):
            info += 0.2
        if state.get('context_keeper_active'):
            info += 0.1
        
        return info
    
    def _min_partition_information(self, state: Dict) -> float:
        """Info de la partition minimale."""
        # Si pas de négociation, partition triviale
        if state.get('negotiation_cycles', 0) == 0:
            return self._total_information(state) * 0.9
        
        # Sinon partition plus difficile
        return self._total_information(state) * 0.4
    
    def _learned_weights(self) -> Dict[str, float]:
        """Poids appris (simplifié)."""
        # Placeholder: à implémenter avec vrai apprentissage
        return {
            'simple': 0.3,
            'lite': 0.4,
            'rigorous': 0.3
        }
    
    def record_performance(self, phi_value: float, actual_quality: float):
        """Enregistre performance pour apprentissage."""
        self.history.append({
            'phi': phi_value,
            'quality': actual_quality,
            'mode': self.mode
        })
        
        # Garder 1000 derniers
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
    
    def set_mode(self, mode: str):
        """Change mode de calcul."""
        if mode in self.MODES:
            self.mode = mode
        else:
            raise ValueError(f"Mode inconnu: {mode}. Choix: {self.MODES}")
    
    def get_mode(self) -> str:
        """Retourne mode actuel."""
        return self.mode

# Instance globale
phi_calculator = PhiCalculator(mode='simple')
