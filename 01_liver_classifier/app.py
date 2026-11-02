import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from liver.schema import FEATURES, NUMERIC, DISCLAIMER, validate
from liver.service import ModelService

ROOT = Path(__file__).resolve().parent

def create_app(artifact_dir=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8192
    try:
        service = ModelService(artifact_dir or ROOT/'artifacts')
        load_error = None
    except (OSError, ValueError, KeyError):
        service, load_error = None, 'Model unavailable. Run download_data.py and train.py, then restart.'
    app.extensions['model_service'] = service

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'"
        return response

    @app.get('/')
    def index():
        return render_template('index.html', numeric=NUMERIC, disclaimer=DISCLAIMER)

    @app.get('/api/health')
    def health():
        return jsonify({'status':'ready' if service else 'unavailable',
                        'model': service.meta['selected_model'] if service else None,
                        'error':load_error}), 200 if service else 503

    @app.post('/api/predict')
    def predict():
        if not service:
            return jsonify(error=load_error),503
        if not request.is_json:
            return jsonify(error='Content-Type must be application/json.'),415
        try:
            frame = validate(request.get_json())
        except ValueError as exc:
            return jsonify(error=str(exc)),400
        return jsonify(service.predict(frame))

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description),exc.code

    @app.errorhandler(Exception)
    def internal_error(exc):
        # Do not log submitted inputs or exception messages that could include them.
        return jsonify(error='Inference failed. Check local artifact compatibility; no prediction was returned.'),500

    return app

if __name__ == '__main__':
    from waitress import serve
    serve(create_app(), host='127.0.0.1', port=int(os.environ.get('PORT','5051')), threads=1)
