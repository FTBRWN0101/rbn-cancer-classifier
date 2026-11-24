import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve,confusion_matrix
from typing import Dict,Any

class ClassificationMetrics:
    @staticmethod
    def evaluate(y_true:np.ndarray,y_pred:np.ndarray,y_proba:np.ndarray|None=None)->Dict[str,Any]:
        results={"accuracy":accuracy_score(y_true,y_pred),"f1_score":f1_score(y_true,y_pred),"confusion_matrix":confusion_matrix(y_true,y_pred)}
        if y_proba is not None:
            results["roc_auc"]=roc_auc_score(y_true,y_proba[:,1])
            fpr,tpr,thresholds=roc_curve(y_true,y_proba[:,1])
            results["roc_curve"]={"fpr":fpr,"tpr":tpr,"thresholds":thresholds}
        return results

