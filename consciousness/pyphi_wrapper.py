"""
PyPhi Wrapper - Calcul Φ exact IIT
Avec fallback gracieux si PyPhi indisponible.
"""

import warnings
warnings.filterwarnings('ignore')

# Try import PyPhi
PYPHI_AVAILABLE = False
try:
    import pyphi
    PYPHI_AVAILABLE = True
except ImportError:
    pyphi = None
    print("⚠️ PyPhi non disponible - utilisera approximation")

import numpy as np
from typing import Dict, Any, Optional

class PyPhiCalculator:
    """Calcul Φ exact avec PyPhi (opt-in strict)."""
    
    def __init__(self):
        self.available = PYPHI_AVAILABLE
        self.max_nodes = 6  # Limite sécurité (au-delà = trop lent)
        
    def compute_exact(self, state: Dict[str, Any], timeout: float = 5.0) -> Optional[float]:
        """
        Calcule Φ exact avec PyPhi.
        
        Args:
            state: État système
            timeout: Timeout secondes (sécurité)
            
        Returns:
            Φ exact ou None si fail/timeout
        """
        if not self.available:
            print("❌ PyPhi non installé")
            return None
        
        try:
            # Extraire réseau
            network = self._build_network(state)
            
            if network is None:
                return None
            
            # Trop grand ?
            if len(network.node_indices) > self.max_nodes:
                print(f"⚠️ Réseau trop grand ({len(network.node_indices)} nœuds > {self.max_nodes})")
                return None
            
            # Calculer Φ
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("PyPhi timeout")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                # État actuel
                current_state = self._get_current_state(state, network)
                
                # Calculer Φ
                sia = pyphi.compute.sia(network, current_state)
                phi = sia.phi
                
                signal.alarm(0)  # Cancel timeout
                
                return float(phi)
                
            except TimeoutError:
                print(f"⚠️ PyPhi timeout après {timeout}s")
                return None
                
        except Exception as e:
            print(f"❌ Erreur PyPhi: {e}")
            return None
    
    def _build_network(self, state: Dict) -> Optional['pyphi.Network']:
        """Construit réseau PyPhi depuis état CRISTAL."""
        if not self.available:
            return None
        
        try:
            # Instances actives
            nodes = []
            if state.get('orchestrator_active'):
                nodes.append(0)
            if state.get('creative_sampler_active'):
                nodes.append(1)
            if state.get('safety_controller_active'):
                nodes.append(2)
            if state.get('context_keeper_active'):
                nodes.append(3)
            
            if len(nodes) < 2:
                return None
            
            n = len(nodes)
            
            # Matrice de transition (simplifiée)
            # Connexions complètes avec poids selon négociation
            tpm = np.zeros((2**n, n))
            
            # Remplir TPM (simplifié)
            for i in range(2**n):
                for j in range(n):
                    # Probabilité activation = fonction état
                    tpm[i, j] = 0.7 if state.get('in_negotiation') else 0.3
            
            # Matrice connectivité
            cm = np.ones((n, n)) - np.eye(n)  # Tous connectés sauf diagonale
            
            # Créer réseau
            network = pyphi.Network(tpm, cm)
            
            return network
            
        except Exception as e:
            print(f"❌ Erreur construction réseau: {e}")
            return None
    
    def _get_current_state(self, state: Dict, network: 'pyphi.Network') -> tuple:
        """État actuel du réseau."""
        # Simplification: tout actif = 1, inactif = 0
        current = []
        
        if state.get('orchestrator_active'):
            current.append(1)
        if state.get('creative_sampler_active'):
            current.append(1)
        if state.get('safety_controller_active'):
            current.append(1)
        if state.get('context_keeper_active'):
            current.append(1)
        
        return tuple(current)
    
    def get_status(self) -> Dict[str, Any]:
        """Statut PyPhi."""
        return {
            'available': self.available,
            'max_nodes': self.max_nodes,
            'library': 'pyphi' if self.available else None
        }

# Instance globale
pyphi_calculator = PyPhiCalculator()
