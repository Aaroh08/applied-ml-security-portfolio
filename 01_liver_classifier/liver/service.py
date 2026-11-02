import json
from pathlib import Path
import joblib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from .schema import FEATURES, NUMERIC, DISCLAIMER, LABELS, encode_raw, decode_raw

class ModelService:
    def __init__(self, artifact_dir):
        root = Path(artifact_dir)
        self.meta = json.loads((root/'metadata.json').read_text())
        # Only load artifacts produced locally by train.py; pickle is not a safe interchange format.
        self.pipeline = joblib.load(root/f"{self.meta['selected_model']}.joblib")
        if list(self.pipeline.feature_names_in_) != FEATURES:
            raise ValueError('Artifact schema does not match application feature order.')
        if self.pipeline['preprocess'].get_feature_names_out().tolist() != self.meta['transformed_feature_order']:
            raise ValueError('Transformed feature names do not match metadata.')
        self.background = np.load(root/'explanation_background.npz')['raw']

    def probability_raw(self, raw):
        return self.pipeline.predict_proba(decode_raw(raw))

    def predict(self, frame, explain=True):
        score = float(self.pipeline.predict_proba(frame)[0,1])
        threshold = self.meta['threshold']
        warnings = []
        for name,(low,high) in self.meta['training_ranges'].items():
            val = frame.iloc[0][name]
            if np.isnan(val):
                warnings.append(f'{name} was imputed using the training median; explanations describe the imputed value.')
            elif val < low or val > high:
                warnings.append(f'{name} is outside the observed training range [{low:g}, {high:g}]. This does not define a medical range.')
        result = {'model': self.meta['selected_model'], 'score': score, 'threshold': threshold,
                  'predicted_label': int(score >= threshold), 'label_description': LABELS[int(score >= threshold)],
                  'disclaimer': DISCLAIMER, 'warnings': warnings, 'calibrated': False}
        if explain:
            raw = encode_raw(frame)
            medians = self.pipeline['preprocess'].named_transformers_['numeric']['imputer'].statistics_
            for j,median in enumerate(medians):
                raw[np.isnan(raw[:,j]),j] = median
            # Explain the entire fitted pipeline in original feature coordinates.
            # Gender remains a single categorical feature, never independent one-hot perturbations.
            np.testing.assert_allclose(self.probability_raw(raw)[0,1], score, atol=1e-12)
            background = self.background[np.random.default_rng(42).choice(len(self.background), min(40,len(self.background)), replace=False)]
            explainer = shap.Explainer(lambda a: self.probability_raw(a)[:,1],
                shap.maskers.Independent(background), algorithm='permutation', feature_names=FEATURES, seed=42)
            values = explainer(raw, max_evals=231, silent=True)
            base = float(values.base_values[0])
            contributions = values.values[0]
            error = abs(base + float(contributions.sum()) - score)
            if error > 1e-6:
                raise ValueError('SHAP reconstruction check failed.')
            result['shap'] = {'method':'Permutation SHAP through the fitted pipeline', 'base_score':base,
                'reconstructed_score': base + float(contributions.sum()), 'additivity_error':error,
                'contributions':[{'feature':n, 'value':float(v)} for n,v in zip(FEATURES,contributions)],
                'note':'Approximate contributions relative to 40 training background rows. Correlated features complicate attribution; these are not causes.'}
            lime = LimeTabularExplainer(self.background, feature_names=FEATURES,
                categorical_features=[len(FEATURES)-1], categorical_names={len(FEATURES)-1:['Female','Male']},
                class_names=LABELS, mode='classification', discretize_continuous=True, random_state=42)
            local = lime.explain_instance(raw[0], self.probability_raw, labels=(1,),
                                          num_features=len(FEATURES), num_samples=2000)
            np.testing.assert_allclose(local.predict_proba[1], score, atol=1e-12)
            approx = float(np.asarray(local.local_pred).ravel()[0])
            result['lime'] = {'contributions':[{'condition':n,'weight':float(v)} for n,v in local.as_list(label=1)],
                'local_weighted_r2':float(local.score), 'surrogate_score':approx,
                'absolute_score_error':abs(approx-score),
                'note':'A local linear approximation, not an exact decomposition. Perturbed biochemical combinations can be implausible. Low R2 or large score error indicates poor fidelity.'}
        return result
