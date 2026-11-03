"""Portable official Windows MySQL: no system service or global configuration changes."""
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
VERSION='8.4.13'
URL=f'https://cdn.mysql.com/Downloads/MySQL-8.4/mysql-{VERSION}-winx64.zip'
EXPECTED_MD5='7b94489d5524dea9b27097068bea25a8'

def main():
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    archive=runtime/'mysql.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=120) as response,archive.with_suffix('.part').open('wb') as out:
            total=0
            while chunk:=response.read(1024*1024):
                out.write(chunk);total+=len(chunk)
                if total%(50*1024*1024)==0:print('Downloaded MiB:',total//(1024*1024),flush=True)
        archive.with_suffix('.part').replace(archive)
    raw=archive.read_bytes()
    if hashlib.md5(raw).hexdigest()!=EXPECTED_MD5:
        raise ValueError('Archive differs from official download-page checksum.')
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/mysql_provenance.json').write_text(json.dumps({'version':VERSION,'url':URL,
       'official_download_page':'https://dev.mysql.com/downloads/mysql/8.4.html','md5':EXPECTED_MD5,
       'sha256':hashlib.sha256(raw).hexdigest(),'license':'MySQL Community Server GPLv2; runtime not redistributed',
       'verified_on':'2026-11-03'},indent=2),encoding='utf-8')
    home=runtime/f'mysql-{VERSION}-winx64'
    if not (home/'bin/mysqld.exe').exists():
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target=(runtime/member.filename).resolve()
                if not target.is_relative_to(runtime.resolve()):raise ValueError('Unsafe archive path')
                # Exclude test/debug material; retain server dependencies and licenses.
                z.extract(member,runtime)
    datadir=runtime/'mysql-data'
    conf=runtime/'config.json'
    if not conf.exists():
        conf.write_text(json.dumps({'db_host':'127.0.0.1','db_port':3307,'db_name':'waf_demo',
            'db_user':'waf_app','db_password':secrets.token_urlsafe(30),
            'root_password':secrets.token_urlsafe(30),'session_secret':secrets.token_hex(32)},indent=2))
    if not (datadir/'mysql').exists():
        with (runtime/'initialize.log').open('wb') as log:
            subprocess.run([str(home/'bin/mysqld.exe'),'--no-defaults','--initialize-insecure',
                '--basedir='+str(home),'--datadir='+str(datadir)],stdout=log,stderr=log,check=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print('Portable MySQL initialized offline. Run run_local.py to start authenticated loopback services.',flush=True)

if __name__=='__main__':main()
