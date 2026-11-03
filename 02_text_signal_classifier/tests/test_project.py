import json
import numpy as np
import pandas as pd
import pytest
from app import Service,create_app
from signals import ROOT,LABELS,clean_text,duplicate_key,mask_label_terms,validate

@pytest.fixture(scope='module')
def service():return Service()

@pytest.mark.parametrize('body',[None,[],{}, {'text':''},{'text':True},{'text':'x'*4001},{'text':'1234567890123'},{'text':'hello world here','user_id':1}])
def test_validation(body):
    with pytest.raises(ValueError):validate(body)

def test_redaction_normalization():
    text=clean_text('  Hello @person contact a@b.com https://example.com  ')
    assert '@person' not in text and 'a@b.com' not in text and 'example.com' not in text
    assert duplicate_key('Hello, WORLD!')==duplicate_key('hello world')
    assert 'bipolar' not in mask_label_terms('A fictional bipolar topic')

def test_split_and_vocabulary(service):
    df=pd.read_csv(ROOT/'data/source.csv')
    splits=json.loads((ROOT/'artifacts/split_manifest.json').read_text())['row_indices']
    keys={k:set(df.loc[ids,'Tweets'].map(duplicate_key)) for k,ids in splits.items()}
    assert not keys['train']&keys['test']
    assert not keys['train']&keys['validation']
    assert not keys['validation']&keys['test']
    analyzer=service.vectorizer.build_analyzer()
    train_terms=set(term for text in df.loc[splits['train'],'Tweets'].map(clean_text) for term in analyzer(text))
    assert set(service.vectorizer.vocabulary_)<=train_terms

def test_frozen_embeddings_and_padding(service):
    assert all(not p.requires_grad for p in service.embedder.model.parameters())
    assert not service.embedder.model.training
    sentence='A fictional ordinary day with friends.'
    a,_=service.embedder.encode([sentence])
    b,_=service.embedder.encode([sentence,'A much longer fictional sentence about a library and its many books and quiet reading rooms.'])
    assert a.shape==(1,768)
    np.testing.assert_allclose(a[0],b[0],atol=2e-6)

def test_truncation(service):
    _,n=service.embedder.encode(['fictional words '*200])
    assert n==1

def test_prediction_and_no_text_persistence(service):
    text=json.loads((ROOT/'examples/fictional_input.json').read_text())['text']
    result=service.predict(clean_text(text))
    assert result['text_only'] is True
    assert 'text' not in result
    for model in result['models'].values():
        assert model['dataset_tag'] in LABELS
        assert sum(model['class_scores'].values())==pytest.approx(1)
    (ROOT/'artifacts/example_response.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def test_api(service):
    c=create_app(service).test_client()
    assert c.get('/').status_code==200
    assert c.get('/api/health').status_code==200
    assert c.post('/api/predict',json={}).status_code==400
    assert c.post('/api/predict',data='test').status_code==415
    assert c.post('/api/predict',data='{',content_type='application/json').status_code==400
    assert c.post('/api/predict',data='x'*22000,content_type='application/json').status_code==413
    response=c.post('/api/predict',json={'text':'A fictional example about a quiet afternoon reading books.'})
    assert response.status_code==200
    assert response.headers['Cache-Control']=='no-store'

def test_missing_models():
    c=create_app(load=False).test_client()
    assert c.get('/api/health').status_code==503
    assert c.post('/api/predict',json={'text':'Fictional test input with enough words.'}).status_code==503

def test_rescore():
    from train import evaluate
    result=json.loads((ROOT/'artifacts/metrics.json').read_text())
    for name,metrics in result.items():
        frame=pd.read_csv(ROOT/f'artifacts/{name}_test_predictions.csv')
        actual=evaluate(frame.label,frame.prediction)
        assert actual['macro_f1']==pytest.approx(metrics['test']['macro_f1'])
        assert actual['confusion_matrix']==metrics['test']['confusion_matrix']
