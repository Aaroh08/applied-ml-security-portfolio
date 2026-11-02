"""One feature contract shared by training, explanations and the API."""
import math
import numpy as np
import pandas as pd

NUMERIC = ['age', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphatase',
           'alt', 'ast', 'total_protein', 'albumin', 'albumin_globulin_ratio']
FEATURES = NUMERIC + ['gender']
CSV_COLUMNS = ['age', 'gender', 'total_bilirubin', 'direct_bilirubin',
               'alkaline_phosphatase', 'alt', 'ast', 'total_protein', 'albumin',
               'albumin_globulin_ratio', 'selector']
LABELS = ['Dataset non-liver-patient label', 'Dataset liver-patient label']
DISCLAIMER = 'Educational dataset-label model. Not a diagnosis, clinical probability, or treatment recommendation.'


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError('Body must be a JSON object with exactly the ten feature fields.')
    missing, extra = sorted(set(FEATURES) - payload.keys()), sorted(payload.keys() - set(FEATURES))
    if missing or extra:
        raise ValueError(f'Missing fields: {missing}; unknown fields: {extra}.')
    clean = {}
    for name in NUMERIC:
        val = payload[name]
        if val is None and name == 'albumin_globulin_ratio':
            clean[name] = np.nan
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f'{name} must be a finite JSON number.')
        if val < 0 or val > 100000:
            raise ValueError(f'{name} must be between 0 and 100000 (software guardrail, not a reference interval).')
        if name == 'age' and (val != int(val) or not 1 <= val <= 90):
            raise ValueError('age must be a whole number from 1 to 90; the dataset top-codes ages above 89 as 90.')
        clean[name] = float(val)
    if payload['gender'] not in ('Female', 'Male'):
        raise ValueError('gender must be Female or Male, the two categories supplied by this dataset.')
    clean['gender'] = payload['gender']
    if clean['direct_bilirubin'] > clean['total_bilirubin']:
        raise ValueError('direct_bilirubin cannot exceed total_bilirubin.')
    if clean['albumin'] > clean['total_protein']:
        raise ValueError('albumin cannot exceed total_protein.')
    return pd.DataFrame([clean], columns=FEATURES)


def encode_raw(frame):
    """Original numeric units plus one binary category, before fitted preprocessing."""
    result = frame[NUMERIC].to_numpy(dtype=float)
    return np.column_stack([result, frame['gender'].map({'Female': 0, 'Male': 1}).to_numpy()])


def decode_raw(array):
    a = np.asarray(array, dtype=float)
    result = pd.DataFrame(a[:, :len(NUMERIC)], columns=NUMERIC)
    if not np.isin(a[:, -1], [0, 1]).all():
        raise ValueError('Explanation category must be 0 or 1.')
    result['gender'] = np.where(a[:, -1] == 0, 'Female', 'Male')
    return result[FEATURES]
