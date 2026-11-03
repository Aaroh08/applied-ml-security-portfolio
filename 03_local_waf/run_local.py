"""Start owned local processes; Ctrl+C stops them. Credentials never enter command arguments."""
import json
import subprocess
import sys
import time
import pymysql
import requests
from storage import ROOT,config,initialize_schema
from setup_mysql import VERSION

def main():
    if not (ROOT/'artifacts/forest.joblib').exists() or not (ROOT/'artifacts/metadata.json').exists():
        raise SystemExit('Train the detector first: python train.py')
    c=config();runtime=ROOT/'runtime';home=runtime/f'mysql-{VERSION}-winx64'
    # File is ignored by Git and used only for local bootstrapping before normal connections.
    bootstrap=runtime/'bootstrap.sql'
    for key in ('root_password','db_password'):
        if not all(ch.isalnum() or ch in '_-' for ch in c[key]):raise ValueError('Unexpected generated credential characters')
    bootstrap.write_text(
        "ALTER USER 'root'@'localhost' IDENTIFIED BY '"+c['root_password']+"';\n"
        "CREATE DATABASE IF NOT EXISTS waf_demo CHARACTER SET utf8mb4;\n"
        "CREATE USER IF NOT EXISTS 'waf_app'@'localhost' IDENTIFIED BY '"+c['db_password']+"';\n"
        "GRANT SELECT, INSERT, UPDATE, CREATE ON waf_demo.* TO 'waf_app'@'localhost';\n",encoding='utf-8')
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0);processes=[];logs=[]
    try:
        log=(runtime/'mysql-server.log').open('ab');logs.append(log)
        dbproc=subprocess.Popen([str(home/'bin/mysqld.exe'),'--no-defaults','--basedir='+str(home),
            '--datadir='+str(runtime/'mysql-data'),'--bind-address=127.0.0.1','--port=3307',
            '--mysqlx=0','--skip-log-bin','--default-time-zone=+00:00','--init-file='+str(bootstrap)],
            stdout=log,stderr=log,creationflags=flags)
        processes.append(dbproc)
        for _ in range(60):
            if dbproc.poll() is not None:raise RuntimeError('MySQL stopped; inspect runtime/mysql-server.log locally.')
            try:
                db=pymysql.connect(host='127.0.0.1',port=3307,user=c['db_user'],password=c['db_password'],database='waf_demo',connect_timeout=1)
                db.close();break
            except pymysql.MySQLError:time.sleep(1)
        else:raise RuntimeError('MySQL was not ready within 60 seconds.')
        bootstrap.unlink(missing_ok=True)
        initialize_schema()
        for role in ['backend','gateway']:
            log=(runtime/f'{role}.log').open('ab');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'app.py',role],cwd=ROOT,stdout=log,stderr=log,creationflags=flags))
        for _ in range(60):
            if any(p.poll() is not None for p in processes):raise RuntimeError('Service startup failed; inspect runtime logs.')
            try:
                requests.get('http://127.0.0.1:5091/health',timeout=1).raise_for_status()
                requests.get('http://127.0.0.1:5053/api/health',timeout=1).raise_for_status()
                break
            except requests.RequestException:time.sleep(1)
        else:raise RuntimeError('Web services did not become ready within 60 seconds.')
        print('Local WAF: http://127.0.0.1:5053 | bundled backend: 127.0.0.1:5091 | MySQL: 127.0.0.1:3307',flush=True)
        print('Keep this terminal open. Ctrl+C stops only these project processes.',flush=True)
        while all(p.poll() is None for p in processes):time.sleep(1)
        raise RuntimeError('A project service stopped; inspect its local runtime log.')
    except KeyboardInterrupt:print('Stopping owned project services.',flush=True)
    finally:
        if processes and processes[0].poll() is None:
            try:
                with pymysql.connect(host='127.0.0.1',port=3307,user='root',password=c['root_password'],connect_timeout=2) as db:
                    db.cursor().execute('SHUTDOWN')
                processes[0].wait(timeout=15)
            except Exception:pass
        for p in reversed(processes):
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill()
        for log in logs:log.close()
        bootstrap.unlink(missing_ok=True)

if __name__=='__main__':main()
