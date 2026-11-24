import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from rbn.simulator import RBNSimulator
from rbn.attractor import AttractorAnalyzer
from rbn.features import AttractorFeatureExtractor

class RBNAttractorClassifier:
    """classify samples using rbn attractor features"""

    def __init__(self,network,base_classifier,handle_imbalance:bool=True,decision_threshold:Optional[float]=None):
        self.simulator=RBNSimulator(network)
        self.analyzer=AttractorAnalyzer()
        self.feature_extractor=AttractorFeatureExtractor()
        self.classifier=base_classifier
        self.handle_imbalance=handle_imbalance
        self.decision_threshold=decision_threshold
        self.scaler=StandardScaler()
        self._is_fitted=False

    def _sample_to_features(self,sample:np.ndarray)->np.ndarray:
        try:
            trajectory=self.simulator.run(sample)
            from rbn.features import CombinedFeatureExtractor
            if isinstance(self.feature_extractor,CombinedFeatureExtractor):
                attractor,attractor_start=self.analyzer.find_attractor_with_transient(trajectory)
                features=self.feature_extractor.extract(trajectory,attractor,attractor_start)
            else:
                attractor=self.analyzer.find_attractor(trajectory)
                features=self.feature_extractor.extract(attractor)
            if features.ndim==0:
                features=np.array([features])
            return features
        except Exception as e:
            print(f"Warning: Feature extraction failed for sample: {e}")
            n_features=getattr(self.feature_extractor,"n_features",8)
            return np.zeros(n_features)

    def fit(self,X:np.ndarray,y:np.ndarray):
        """fit base classifier on extracted attractor features"""
        print("  Extracting attractor features from training data...")
        features_list=[]
        for i,sample in enumerate(X):
            if (i+1)%50==0:
                print(f"    Processed {i+1}/{len(X)} samples...")
            features_list.append(self._sample_to_features(sample))
        features=np.vstack(features_list)
        print(f"  ✓ Feature extraction complete")
        print(f"    Feature matrix shape: {features.shape}")
        print(f"    Feature range: [{features.min():.3f}, {features.max():.3f}]")
        feature_stds=features.std(axis=0)
        if np.any(feature_stds==0):
            print(f"    Warning: {np.sum(feature_stds == 0)} constant features detected")
        features_scaled=self.scaler.fit_transform(features)
        fit_params={}
        if self.handle_imbalance:
            if hasattr(self.classifier,"class_weight"):
                self.classifier.class_weight="balanced"
                print("    Using class_weight='balanced'")
            else:
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weights=compute_sample_weight("balanced",y)
                fit_params["sample_weight"]=sample_weights
                print("    Using computed sample weights")
        print(f"  Training base classifier...")
        self.classifier.fit(features_scaled,y,**fit_params)
        self._is_fitted=True
        return self

    def predict(self,X:np.ndarray)->np.ndarray:
        """predict labels from binary input"""
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        if self.decision_threshold is not None:
            y_proba=self.predict_proba(X)
            return (y_proba[:,1]>=self.decision_threshold).astype(int)
        features=np.vstack([self._sample_to_features(sample) for sample in X])
        features_scaled=self.scaler.transform(features)
        return self.classifier.predict(features_scaled)

    def predict_proba(self,X:np.ndarray)->np.ndarray:
        """predict class probabilities from binary input"""
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        features=np.vstack([self._sample_to_features(sample) for sample in X])
        features_scaled=self.scaler.transform(features)
        return self.classifier.predict_proba(features_scaled)

    def optimize_threshold(self,X_val:np.ndarray,y_val:np.ndarray,metric:str="f1",min_threshold:float=0.2,max_threshold:float=0.8)->float:
        """pick a decision threshold on validation data"""
        from sklearn.metrics import f1_score,balanced_accuracy_score,matthews_corrcoef
        print("  Finding optimal decision threshold...")
        y_proba=self.predict_proba(X_val)[:,1]
        thresholds=np.linspace(min_threshold,max_threshold,50)
        best_score=-np.inf
        best_threshold=0.5
        for threshold in thresholds:
            y_pred=(y_proba>=threshold).astype(int)
            n_class_0=np.sum(y_pred==0)
            n_class_1=np.sum(y_pred==1)
            if n_class_0==0 or n_class_1==0:
                continue
            if metric=="f1":
                score=f1_score(y_val,y_pred,zero_division=0)
            elif metric=="balanced_accuracy":
                score=balanced_accuracy_score(y_val,y_pred)
            elif metric=="mcc":
                score=matthews_corrcoef(y_val,y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            if score>best_score:
                best_score=score
                best_threshold=threshold
        print(f"  ✓ Optimal threshold: {best_threshold:.3f} ({metric}={best_score:.3f})")
        self.decision_threshold=best_threshold
        return best_threshold
