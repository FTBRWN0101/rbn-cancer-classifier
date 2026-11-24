import numpy as np
from typing import Dict,List
from .metrics import ClassificationMetrics

class NoiseRobustnessEvaluator:
    def __init__(self,classifier):
        self.classifier=classifier

    @staticmethod
    def _apply_noise(X:np.ndarray,noise_level:float)->np.ndarray:
        noisy_X=X.copy()
        flip_mask=np.random.rand(*X.shape)<noise_level
        noisy_X[flip_mask]^=1
        return noisy_X

    def evaluate(self,X:np.ndarray,y:np.ndarray,noise_levels:List[float])->Dict[float,Dict]:
        results={}
        for noise in noise_levels:
            X_noisy=self._apply_noise(X,noise)
            y_pred=self.classifier.predict(X_noisy)
            y_proba=None
            if hasattr(self.classifier,"predict_proba"):
                y_proba=self.classifier.predict_proba(X_noisy)
            metrics=ClassificationMetrics.evaluate(y_true=y,y_pred=y_pred,y_proba=y_proba)
            results[noise]=metrics
        return results
