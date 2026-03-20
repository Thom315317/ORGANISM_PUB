"""
Consciousness Theory Testbed
Orchestre les 7 théories et mesure performances.
"""

from typing import Dict, Any, List
import json
import os
from datetime import datetime

from consciousness.theories.iit import IITImplementation
from consciousness.theories.gwt import GlobalWorkspace
from consciousness.theories.fep import FreeEnergyPrinciple
from consciousness.theories.hot import HigherOrderThought
from consciousness.theories.rpt import RecurrentProcessing
from consciousness.theories.dyn import DynamicalSynchrony
from consciousness.theories.hybrid import AdaptiveHybrid

class ConsciousnessTestbed:
    """Testbed expérimental des théories de orchestrator."""
    
    def __init__(self, data_dir='data/consciousness_experiments'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Instancier théories
        self.theories = {
            'IIT': IITImplementation(),
            'GWT': GlobalWorkspace(),
            'FEP': FreeEnergyPrinciple(),
            'HOT': HigherOrderThought(),
            'RPT': RecurrentProcessing(),
            'DYN': DynamicalSynchrony(),
            'HYBRID': AdaptiveHybrid()
        }
        
        # Tracking performances
        self.performance_data = {name: [] for name in self.theories}
        self.experiment_log = []
    
    def measure_all(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Mesure orchestrator selon toutes les théories."""
        scores = {}
        
        # Mesurer chaque théorie
        for name, theory in self.theories.items():
            if name == 'HYBRID':
                # HYBRID a besoin des autres scores
                continue
            try:
                scores[name] = theory.compute(state)
            except Exception as e:
                print(f"Erreur {name}: {e}")
                scores[name] = 0.0
        
        # HYBRID combine les autres
        try:
            scores['HYBRID'] = self.theories['HYBRID'].compute(state, scores)
        except:
            scores['HYBRID'] = sum(scores.values()) / len(scores) if scores else 0.0
        
        return scores
    
    def predict_quality(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Prédit qualité de décision selon chaque théorie."""
        consciousness_scores = self.measure_all(state)
        
        # Hypothèse simple: consciousness score ≈ quality prediction
        predictions = {}
        for name, score in consciousness_scores.items():
            predictions[name] = score
        
        return predictions
    
    def record_outcome(self, state: Dict, predictions: Dict[str, float], actual_quality: float):
        """Enregistre résultat réel et met à jour performances."""
        # Calculer erreurs
        errors = {}
        for theory, predicted in predictions.items():
            error = abs(predicted - actual_quality)
            errors[theory] = error
            self.performance_data[theory].append(error)
        
        # Mettre à jour HYBRID
        if len(self.performance_data['IIT']) > 10:  # Minimum données
            avg_errors = {
                name: sum(errs[-100:]) / min(100, len(errs))
                for name, errs in self.performance_data.items()
                if name != 'HYBRID'
            }
            self.theories['HYBRID'].update_weights(avg_errors)
        
        # Logger expérience
        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'actual_quality': actual_quality,
            'errors': errors
        })
        
        # Sauvegarder périodiquement
        if len(self.experiment_log) % 100 == 0:
            self.save_results()
    
    def get_best_theory(self) -> str:
        """Retourne théorie avec meilleures prédictions."""
        avg_errors = {}
        for name, errors in self.performance_data.items():
            if errors:
                avg_errors[name] = sum(errors) / len(errors)
        
        if not avg_errors:
            return 'HYBRID'
        
        return min(avg_errors, key=avg_errors.get)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Stats complètes du testbed."""
        stats = {
            'total_experiments': len(self.experiment_log),
            'theories': {}
        }
        
        for name, errors in self.performance_data.items():
            if errors:
                stats['theories'][name] = {
                    'avg_error': sum(errors) / len(errors),
                    'min_error': min(errors),
                    'max_error': max(errors),
                    'samples': len(errors)
                }
        
        # Ranking
        if stats['theories']:
            ranking = sorted(
                stats['theories'].items(),
                key=lambda x: x[1]['avg_error']
            )
            stats['ranking'] = [name for name, _ in ranking]
            stats['best_theory'] = ranking[0][0]
        
        return stats
    
    def save_results(self):
        """Sauvegarde résultats expériences."""
        # Stats
        stats_file = os.path.join(self.data_dir, 'statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
        
        # Log complet
        log_file = os.path.join(self.data_dir, 'experiments.jsonl')
        with open(log_file, 'w') as f:
            for exp in self.experiment_log:
                f.write(json.dumps(exp) + '\n')
        
        print(f"✓ Résultats sauvegardés: {len(self.experiment_log)} expériences")

# Instance globale
consciousness_testbed = ConsciousnessTestbed()
