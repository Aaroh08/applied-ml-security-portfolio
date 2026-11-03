import json
import os
import joblib
import numpy as np
from flask import Flask,jsonify,render_template,request
from werkzeug.exceptions import HTTPException
from signals import ROOT,DISCLAIMER,validate,Embedder

class Service:
    def __init__(self):
        self.meta=json.loads((ROOT/'artifacts/metadata.json').read_text())
        self.vectorizer=joblib.load(ROOT/'artifacts/vectorizer.joblib')
        self.models={name:joblib.load(ROOT/f'artifacts/{name}.joblib') for name in ['tfidf','distilbert']}
        self.embedder=Embedder()

    def predict(self,text):
        vectors=self.vectorizer.transform([text])
        embeddings,truncated=self.embedder.encode([text])
        result={'selected_model':self.meta['selected_model'],'disclaimer':DISCLAIMER,
                'text_only':True,'truncated_at_128_tokens':bool(truncated),'models':{},
                'warning':'Dataset tags are lexicon-derived topics, not confirmed conditions. The no tag does not rule out distress.'}
        for name,features in [('tfidf',vectors),('distilbert',embeddings)]:
            model=self.models[name]; probs=model.predict_proba(features)[0]; index=int(probs.argmax())
            result['models'][name]={'dataset_tag':str(model.classes_[index]),'max_score':float(probs[index]),
                'class_scores':{str(k):float(v) for k,v in zip(model.classes_,probs)},'calibrated':False}
        result['models_disagree']=result['models']['tfidf']['dataset_tag']!=result['models']['distilbert']['dataset_tag']
        result['tfidf_known_feature_count']=int(vectors.nnz)
        return result

def create_app(service=None,load=True):
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=20000
    if service is None and load:
        try: service=Service()
        except (OSError,ValueError,KeyError): service=None
    app.extensions['model_service']=service
    @app.after_request
    def secure(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
        return response
    @app.get('/')
    def index():return render_template('index.html')
    @app.get('/api/health')
    def health():return jsonify(status='ready' if service else 'unavailable',text_only=True),200 if service else 503
    @app.post('/api/predict')
    def predict():
        if not service:return jsonify(error='Models unavailable. Run download.py and train.py, then restart.'),503
        if not request.is_json:return jsonify(error='Content-Type must be application/json.'),415
        try:text=validate(request.get_json())
        except ValueError as exc:return jsonify(error=str(exc)),400
        return jsonify(service.predict(text))
    @app.errorhandler(HTTPException)
    def http_error(exc):return jsonify(error=exc.description),exc.code
    @app.errorhandler(Exception)
    def error(exc):return jsonify(error='Inference failed; no classification was returned.'),500
    return app

if __name__=='__main__':
    from waitress import serve
    serve(create_app(),host='127.0.0.1',port=int(os.environ.get('PORT','5052')),threads=1)
