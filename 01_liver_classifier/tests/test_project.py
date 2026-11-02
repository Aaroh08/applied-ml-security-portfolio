import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app import create_app
from liver.schema import validate, FEATURES, NUMERIC, CSV_COLUMNS, encode_raw, decode_raw
from liver.service import ModelService

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def sample():
    return json.loads((ROOT/'examples/fictional_input.json').read_text())

@pytest.fixture(scope='module')
def service():
    return ModelService(ROOT/'artifacts')

@pytest.mark.parametrize('field,value', [('age',True),('age',45.5),('age',91),('alt',float('nan')),('ast',-1),('gender','Unknown'),('total_protein','6.8'),('albumin',20)])
def test_invalid_values(sample,field,value):
    sample[field]=value
    with pytest.raises(ValueError): validate(sample)

def test_unknown_missing_and_bad_body(sample):
    for body in [[],None,{**sample,'selector':1},{k:v for k,v in sample.items() if k!='alt'}]:
        with pytest.raises(ValueError): validate(body)

def test_raw_encoding_and_pipeline_match(sample,service):
    frame=validate(sample)
    pd.testing.assert_frame_equal(frame,decode_raw(encode_raw(frame)))
    np.testing.assert_allclose(service.pipeline.predict_proba(frame),service.probability_raw(encode_raw(frame)))
    assert service.pipeline['preprocess'].get_feature_names_out().tolist()==NUMERIC+['gender_Male']

def test_training_only_and_disjoint_profiles(service):
    df=pd.read_csv(ROOT/'data/ilpd.csv',header=None,names=CSV_COLUMNS)
    splits=json.loads((ROOT/'artifacts/split_manifest.json').read_text())['row_indices']
    hashes={k:set(pd.util.hash_pandas_object(df.loc[ids,FEATURES],index=False)) for k,ids in splits.items()}
    assert not hashes['train'] & hashes['test']
    assert not hashes['train'] & hashes['validation']
    assert not hashes['test'] & hashes['validation']
    expected=df.loc[splits['train'],NUMERIC].median().to_numpy()
    actual=service.pipeline['preprocess'].named_transformers_['numeric']['imputer'].statistics_
    np.testing.assert_allclose(expected,actual)

def test_explanations_reconstruct_actual_score(sample,service):
    result=service.predict(validate(sample))
    assert 0<=result['score']<=1
    assert result['shap']['additivity_error']<1e-6
    assert {x['feature'] for x in result['shap']['contributions']}==set(FEATURES)
    assert np.isfinite(result['lime']['local_weighted_r2'])
    assert len(result['lime']['contributions'])==10
    (ROOT/'artifacts/example_response.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def test_imputation_and_out_of_distribution(sample,service):
    sample['albumin_globulin_ratio']=None
    sample['alt']=99999
    result=service.predict(validate(sample),explain=False)
    assert any('imputed' in x for x in result['warnings'])
    assert any('outside' in x for x in result['warnings'])

def test_api_errors_and_health(sample):
    client=create_app().test_client()
    assert client.get('/api/health').status_code==200
    assert client.get('/').status_code==200
    assert client.post('/api/predict',data='hello').status_code==415
    assert client.post('/api/predict',json={}).status_code==400
    assert client.post('/api/predict',data='{',content_type='application/json').status_code==400
    assert client.post('/api/predict',data='x'*9000,content_type='application/json').status_code==413
    assert client.get('/api/predict').status_code==405

def test_missing_artifact_service(tmp_path,sample):
    client=create_app(tmp_path).test_client()
    assert client.get('/api/health').status_code==503
    assert client.post('/api/predict',json=sample).status_code==503

def test_api_prediction_and_privacy_headers(sample):
    response=create_app().test_client().post('/api/predict',json=sample)
    assert response.status_code==200
    assert response.json['calibrated'] is False
    assert response.headers['Cache-Control']=='no-store'
    assert 'input' not in response.json

def test_saved_metrics_match_predictions():
    from train import metrics
    saved=json.loads((ROOT/'artifacts/metrics.json').read_text())
    selection=json.loads((ROOT/'artifacts/selection.json').read_text())
    winner=max(saved,key=lambda n:(saved[n]['validation']['macro_f1'],saved[n]['validation']['average_precision']))
    assert winner==selection['selected_model']
    for name,result in saved.items():
        df=pd.read_csv(ROOT/f'artifacts/{name}_test_predictions.csv')
        rescored=metrics(df.label,df.score,result['test']['threshold'])
        assert rescored['confusion_matrix']==result['test']['confusion_matrix']
        assert rescored['roc_auc']==pytest.approx(result['test']['roc_auc'])
