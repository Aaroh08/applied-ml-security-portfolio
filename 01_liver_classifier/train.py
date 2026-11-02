"""Fixed, auditable experiment: deduplicate, split, fit, select on validation, test."""
import hashlib
import json
import platform
import time
from pathlib import Path
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, auc, f1_score, balanced_accuracy_score,
    brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)
from liver.schema import FEATURES, NUMERIC, CSV_COLUMNS, encode_raw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
SEED = 42

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def preprocess():
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('gender', OneHotEncoder(categories=[['Female', 'Male']], drop='if_binary',
                                handle_unknown='error', sparse_output=False), ['gender'])
    ], verbose_feature_names_out=False)

def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    p, r, _ = precision_recall_curve(y, scores)
    return {'threshold': float(threshold), 'n': len(y), 'positive_count': int(sum(y)),
            'classification_report': classification_report(y, pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            'roc_auc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'pr_auc_trapezoidal': float(auc(r, p)),
            'macro_f1': float(f1_score(y, pred, average='macro')),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'brier_score': float(brier_score_loss(y, scores))}

def choose_threshold(y, scores):
    candidates = np.linspace(0.1, 0.9, 81)
    # Balance both classes, then prefer thresholds closest to 0.5, then lower value.
    return float(max(candidates, key=lambda t: (f1_score(y, scores >= t, average='macro'), -abs(t-.5), -t)))

def main():
    OUT.mkdir(exist_ok=True)
    csv = ROOT / 'data/ilpd.csv'
    provenance = json.loads((OUT / 'data_provenance.json').read_text())
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == provenance['csv_sha256']
    df = pd.read_csv(csv, header=None, names=CSV_COLUMNS)
    assert set(df.selector.unique()) == {1, 2}
    assert set(df.gender.unique()) == {'Male', 'Female'}
    audit = {'raw_rows': len(df), 'raw_labels': {str(k): int(v) for k,v in df.selector.value_counts().items()},
             'missing': {k: int(v) for k,v in df.isna().sum().items()},
             'exact_duplicate_rows': int(df.duplicated().sum()),
             'feature_duplicate_rows': int(df.duplicated(FEATURES).sum()),
             'source_discrepancies': 'UCI text says 584 once and 583 elsewhere, and says no missing values. Actual CSV counts take precedence.',
             'patient_id_available': False, 'units': 'UCI specifies years for age; biochemical units are not specified. Do not assume units for real-world input.',
             'direct_exceeds_total': int((df.direct_bilirubin > df.total_bilirubin).sum()),
             'albumin_exceeds_protein': int((df.albumin > df.total_protein).sum())}
    # Remove *all* conflicting profiles; collapse identical profiles otherwise.
    keys = pd.util.hash_pandas_object(df[FEATURES], index=False)
    conflicting = df.groupby(keys).selector.nunique()
    conflict_keys = conflicting[conflicting > 1].index
    audit['conflicting_profiles'] = len(conflict_keys)
    audit['conflicting_rows_removed'] = int(keys.isin(conflict_keys).sum())
    df = df.loc[~keys.isin(conflict_keys)].drop_duplicates(FEATURES).copy()
    df['label'] = (df.selector == 1).astype(int)
    train_ids, rest = train_test_split(df.index, test_size=.4, random_state=SEED, stratify=df.label)
    val_ids, test_ids = train_test_split(rest, test_size=.5, random_state=SEED, stratify=df.loc[rest,'label'])
    splits = {'train': train_ids.tolist(), 'validation': val_ids.tolist(), 'test': test_ids.tolist()}
    audit['clean_rows'] = len(df)
    audit['split_counts'] = {k: {'n': len(ids), 'positive': int(df.loc[ids,'label'].sum())} for k,ids in splits.items()}
    audit['training_numeric_summary'] = json.loads(df.loc[train_ids,NUMERIC].describe().to_json())
    audit['training_correlations'] = json.loads(df.loc[train_ids,NUMERIC].corr().to_json())
    audit['leakage_review'] = 'Selector is excluded from features. No IDs/timestamps are supplied. Identical feature profiles cannot cross splits. Related records with different features remain undetectable. Correlated biochemical variables are retained; they complicate explanation attribution.'
    dump('data_audit.json', audit)
    dump('split_manifest.json', {'seed': SEED, 'csv_sha256': provenance['csv_sha256'], 'row_indices': splits})
    X, y = df[FEATURES], df.label
    models = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic': LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, random_state=SEED),
        'forest': RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=5,
                                         class_weight='balanced', random_state=SEED, n_jobs=1)
    }
    fitted, validation, timings = {}, {}, {}
    for name, model in models.items():
        pipe = Pipeline([('preprocess', preprocess()), ('model', model)])
        start = time.perf_counter()
        pipe.fit(X.loc[train_ids], y.loc[train_ids])
        timings[name] = time.perf_counter() - start
        scores = pipe.predict_proba(X.loc[val_ids])[:,1]
        threshold = .5 if name == 'dummy' else choose_threshold(y.loc[val_ids], scores)
        validation[name] = metrics(y.loc[val_ids], scores, threshold)
        fitted[name] = pipe
    # Include the dummy in selection rather than hiding a baseline win.
    selected = max(models, key=lambda n: (validation[n]['macro_f1'], validation[n]['average_precision']))
    dump('selection.json', {'selected_model': selected, 'criterion': 'validation macro-F1; tie-break average precision',
         'threshold_criterion': 'validation macro-F1 on 0.10..0.90 grid in steps of 0.01',
         'threshold': validation[selected]['threshold'], 'validation': validation})
    results = {}
    figure, axes = plt.subplots(1, 2, figsize=(10,4))
    for name, pipe in fitted.items():
        scores = pipe.predict_proba(X.loc[test_ids])[:,1]
        threshold = validation[name]['threshold']
        results[name] = {'validation': validation[name], 'test': metrics(y.loc[test_ids], scores, threshold),
                         'test_at_0_5': metrics(y.loc[test_ids], scores, .5), 'fit_seconds': timings[name]}
        pd.DataFrame({'source_row': test_ids, 'label': y.loc[test_ids].to_numpy(),
                      'score': scores, 'prediction': (scores >= threshold).astype(int)}).to_csv(OUT/f'{name}_test_predictions.csv', index=False)
        joblib.dump(pipe, OUT / f'{name}.joblib')
        RocCurveDisplay.from_predictions(y.loc[test_ids], scores, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y.loc[test_ids], scores, name=name, ax=axes[1])
    figure.tight_layout(); figure.savefig(OUT/'test_curves.png', dpi=150); plt.close(figure)
    fig, axes = plt.subplots(1,3,figsize=(11,3))
    for ax,(name,result) in zip(axes,results.items()):
        ConfusionMatrixDisplay(np.array(result['test']['confusion_matrix']), display_labels=['Non-liver','Liver']).plot(ax=ax,colorbar=False)
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(OUT/'confusion_matrices.png',dpi=150); plt.close(fig)
    selected_pipe = fitted[selected]
    train_raw = encode_raw(X.loc[train_ids])
    # Background is training-only, filled with the exact pipeline medians.
    medians = selected_pipe['preprocess'].named_transformers_['numeric']['imputer'].statistics_
    for j,median in enumerate(medians):
        train_raw[np.isnan(train_raw[:,j]),j] = median
    np.savez_compressed(OUT/'explanation_background.npz', raw=train_raw)
    dump('metadata.json', {'selected_model': selected, 'threshold': validation[selected]['threshold'],
        'feature_order': FEATURES, 'transformed_feature_order': selected_pipe['preprocess'].get_feature_names_out().tolist(),
        'seed': SEED, 'trained_rows': len(train_ids), 'csv_sha256': provenance['csv_sha256'],
        'training_ranges': {n: [float(X.loc[train_ids,n].min()),float(X.loc[train_ids,n].max())] for n in NUMERIC},
        'models': list(models), 'prediction_target': 'ILPD Selector=1 versus Selector=2',
        'calibrated': False, 'test_selected_by_validation_only': True})
    dump('metrics.json', results)
    dump('environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {n: importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','Flask','shap','lime']}})
    print(json.dumps({'selected_model':selected,'audit':audit['split_counts'], 'test': results[selected]['test']},indent=2))

if __name__ == '__main__':
    main()
