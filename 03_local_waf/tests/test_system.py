import json
import re
from types import SimpleNamespace
import pandas as pd
import pytest
import requests
from app import create_gateway
from detector import rule_hits, canonicalize
from train import generate, TEMPLATES

class MemoryStore:
    def __init__(self):self.events=[]
    def begin(self,e):self.events.append(e);return 1
    def finish(self,*args):pass
    def health(self):pass
    def recent(self):return {'events':self.events}

@pytest.mark.parametrize('payload',["1 UNION SELECT name",'%253Cscript%253Ealert(1)%253C/script%253E','<img onerror=alert(1)>','javascript:alert(1)'])
def test_encoded_patterns(payload):assert rule_hits(payload)

def test_template_split():
    df=generate()
    assert not df.payload.duplicated().any()
    assert df.groupby('family').split.nunique().max()==1
    assert len(df)==930

def test_block_and_redaction():
    store=MemoryStore();calls=[]
    client=create_gateway(store=store,upstream=lambda **kw:calls.append(kw)).test_client()
    r=client.get('/app/search',query_string={'q':"SECRET 1 UNION SELECT name"})
    assert r.status_code==403 and not calls
    assert 'SECRET' not in json.dumps(store.events)

def test_fixed_upstream_and_errors():
    calls=[]
    def forward(**kw):calls.append(kw);return SimpleNamespace(content=b'{}',status_code=200,headers={'Content-Type':'application/json'})
    client=create_gateway(store=MemoryStore(),upstream=forward).test_client()
    assert client.get('/app/search?q=notebook').status_code==200
    assert calls[0]['url']=='http://127.0.0.1:5091/search'
    assert client.get('/app/http://example.com').status_code==404
    assert client.post('/app/note',data='x',content_type='text/plain').status_code==415
    assert client.post('/app/note',data='x'*9000,content_type='application/json').status_code==413
    assert client.get('/',headers={'Host':'evil.example'}).status_code==403

def test_fail_closed():
    class Broken(MemoryStore):
        def begin(self,e):raise OSError('offline')
    calls=[]
    client=create_gateway(store=Broken(),upstream=lambda **kw:calls.append(kw)).test_client()
    assert client.get('/app/search?q=notebook').status_code==503 and not calls

def test_upstream_failure():
    def offline(**kw):raise requests.ConnectionError()
    client=create_gateway(store=MemoryStore(),upstream=offline).test_client()
    assert client.get('/app/search?q=notebook').status_code==502

@pytest.mark.integration
def test_live_mysql_and_layers():
    base='http://127.0.0.1:5053';backend='http://127.0.0.1:5091'
    s=requests.Session()
    assert s.get(base+'/api/health',timeout=5).json()['database']=='mysql'
    assert s.get(base+'/app/search',params={'q':'notebook'},timeout=5).json()['count']==1
    for q in ["1' UNION SELECT name FROM products --",'<script>alert(1)</script>']:
        assert s.get(base+'/app/search',params={'q':q},timeout=5).status_code==403
    assert s.post(base+'/app/note',data={'note':'sample note'},timeout=5).status_code==400
    page=s.get(base+'/app/',timeout=5)
    assert page.status_code==200
    token=re.search('name="csrf_token" value="([^"]+)"',page.text).group(1)
    assert s.post(base+'/app/note',data={'note':'sample note','csrf_token':token},timeout=5).status_code==200
    # Direct loopback backend checks prove application defenses independently of WAF.
    assert s.get(backend+'/search',params={'q':"' OR 1=1 --"},timeout=5).json()['count']==0
    assert s.get(backend+'/search',timeout=5).json()['count']==3
    page=s.get(backend+'/',timeout=5)
    token=re.search('name="csrf_token" value="([^"]+)"',page.text).group(1)
    result=s.post(backend+'/note',data={'note':'<script>fictional</script>','csrf_token':token},timeout=5)
    assert result.status_code==200 and '&lt;script&gt;' in result.text and '<script>' not in result.text
    events=s.get(base+'/api/events',timeout=5).json()
    assert events['summary']['blocked']>=2
    assert all(e['request_summary']['payload']=='[REDACTED]' for e in events['events'])
    assert '<script>fictional' not in json.dumps(events)
