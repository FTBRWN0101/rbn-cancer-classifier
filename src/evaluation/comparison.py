import numpy as np
from typing import Dict
from .metrics import ClassificationMetrics

class ModelComparison:
    @staticmethod
    def compare(model_a,model_b,X_test:np.ndarray,y_test:np.ndarray,label_a:str="model_a",label_b:str="model_b")->Dict[str,Dict]:
        results={}
        for model,label in zip([model_a,model_b],[label_a,label_b]):
            y_pred=model.predict(X_test)
            y_proba=None
            if hasattr(model,"predict_proba"):
                y_proba=model.predict_proba(X_test)
            results[label]=ClassificationMetrics.evaluate(y_true=y_test,y_pred=y_pred,y_proba=y_proba)
        return results
