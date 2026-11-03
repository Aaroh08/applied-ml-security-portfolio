import argparse
import json
import time
import uuid
from pathlib import Path
import joblib
import requests
from flask import Flask,jsonify,render_template,request,Response
from flask_wtf.csrf import CSRFProtect,CSRFError
from werkzeug.exceptions import HTTPException
from detector import Detector
from storage import ROOT,config,connection,EventStore

def harden(app):
    app.config.update(MAX_CONTENT_LENGTH=8192,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict',
                      SESSION_COOKIE_SECURE=False) # HTTP loopback demo; HTTPS is required for Secure cookies.
    @app.before_request
    def local_only():
        if request.remote_addr not in ('127.0.0.1','::1') or request.host.split(':')[0] not in ('127.0.0.1','localhost'):
            return jsonify(error='Localhost demo only.'),403
    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; form-action 'self'"
        return response
    @app.errorhandler(HTTPException)
    def http_error(exc):return jsonify(error=exc.description),exc.code
    @app.errorhandler(Exception)
    def error(exc):return jsonify(error='Local service error; check service health.'),503

def create_backend():
    app=Flask('demo_backend',template_folder=str(ROOT/'templates'),static_folder=str(ROOT/'static'))
    harden(app);app.secret_key=config()['session_secret'];CSRFProtect(app)
    @app.errorhandler(CSRFError)
    def csrf_error(exc):return jsonify(error='CSRF validation failed. Reload the form for a valid session token.'),400
    @app.get('/health')
    def health():return jsonify(status='ready')
    @app.get('/')
    def index():
        with connection() as db,db.cursor() as cur:
            cur.execute('SELECT body FROM notes ORDER BY id DESC LIMIT 5');notes=cur.fetchall()
        return render_template('demo.html',notes=notes)
    @app.get('/search')
    def search():
        q=request.args.get('q','')
        if len(q)>200:return jsonify(error='Search is limited to 200 characters.'),400
        with connection() as db,db.cursor() as cur:
            cur.execute('SELECT id,name,price FROM products WHERE name LIKE %s ORDER BY id',('%'+q+'%',))
            rows=cur.fetchall()
        return jsonify(products=rows,count=len(rows))
    @app.post('/note')
    def note():
        body=request.form.get('note','').strip()
        if not 1<=len(body)<=200:return jsonify(error='Fictional note must have 1–200 characters.'),400
        with connection() as db,db.cursor() as cur:
            cur.execute('INSERT INTO notes(body) VALUES (%s)',(body,))
        return render_template('saved.html',note=body)
    return app

def create_gateway(store=None,detector=None,upstream=None):
    app=Flask('gateway',template_folder=str(ROOT/'templates'),static_folder=str(ROOT/'static'))
    harden(app);store=store or EventStore()
    meta=json.loads((ROOT/'artifacts/metadata.json').read_text())
    detector=detector or Detector(joblib.load(ROOT/'artifacts/forest.joblib'),meta['threshold'])
    upstream=upstream or requests.request
    @app.get('/')
    def dashboard():return render_template('dashboard.html',threshold=meta['threshold'])
    @app.get('/api/health')
    def health():
        try:store.health()
        except Exception:return jsonify(status='unavailable',database='unavailable'),503
        return jsonify(status='ready',database='mysql',threshold=meta['threshold'])
    @app.get('/api/events')
    def events():return jsonify(store.recent())

    @app.route('/app/',defaults={'path':''},methods=['GET','POST'])
    @app.route('/app/<path:path>',methods=['GET','POST'])
    def proxy(path):
        if path not in ('','search','note'):return jsonify(error='Only bundled demo routes are permitted.'),404
        if request.method=='POST' and request.mimetype not in ('application/x-www-form-urlencoded','application/json'):
            return jsonify(error='Only JSON or URL-encoded form bodies are supported.'),415
        body=request.get_data() # Flask enforces the configured size limit.
        values=[path]
        values.extend(str(v) for k in request.args for v in request.args.getlist(k))
        if request.method=='POST':
            if request.is_json:
                payload=request.get_json()
                if not isinstance(payload,dict):return jsonify(error='JSON body must be an object.'),400
                # Recursively serialized data stays bounded by MAX_CONTENT_LENGTH.
                values.append(json.dumps({k:v for k,v in payload.items() if k!='csrf_token'}))
            else:values.extend(str(v) for k in request.form if k!='csrf_token' for v in request.form.getlist(k))
        decision=detector.inspect(' '.join(values))
        rid=uuid.uuid4().hex;started=time.perf_counter()
        # Entire bodies, query values, cookies, auth and token headers are omitted, not partially masked.
        summary={'payload':'[REDACTED]','query':'[REDACTED]','body_bytes':len(body),
                 'authorization':'[REDACTED]','cookie':'[REDACTED]','csrf_token':'[REDACTED]'}
        try:
            event_id=store.begin({'request_id':rid,'method':request.method,'route':'/app/'+path,
                'decision':'block' if decision['blocked'] else 'allow','reason':decision['reason'],
                'model_score':decision['model_score'],'rule_hits':decision['rule_hits'],'request_summary':summary})
        except Exception:return jsonify(error='Event database unavailable; request was not forwarded.'),503
        if decision['blocked']:
            response=jsonify(error='Request blocked by local prototype.',request_id=rid,**decision);response.status_code=403
        else:
            headers={k:request.headers[k] for k in ['Content-Type','Cookie','X-CSRFToken'] if k in request.headers}
            try:
                result=upstream(method=request.method,url='http://127.0.0.1:5091/'+path,
                    params=list(request.args.items(multi=True)),data=body,headers=headers,timeout=3,allow_redirects=False)
                response=Response(result.content,status=result.status_code)
                for key in ['Content-Type','Set-Cookie']:
                    if key in result.headers:response.headers[key]=result.headers[key]
            except requests.RequestException:
                response=jsonify(error='Bundled upstream unavailable.',request_id=rid);response.status_code=502
        response.headers['X-Request-ID']=rid
        try:store.finish(event_id,response.status_code,(time.perf_counter()-started)*1000)
        except Exception:response.headers['X-Event-Completion']='unavailable'
        return response
    return app

if __name__=='__main__':
    from waitress import serve
    parser=argparse.ArgumentParser();parser.add_argument('role',choices=['gateway','backend']);args=parser.parse_args()
    serve(create_gateway() if args.role=='gateway' else create_backend(),host='127.0.0.1',
          port=5053 if args.role=='gateway' else 5091,threads=4)
