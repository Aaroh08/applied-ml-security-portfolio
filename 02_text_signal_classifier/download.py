import hashlib
import json
from pathlib import Path
import urllib.request
import time

ROOT=Path(__file__).resolve().parent
DATA_URL='https://zenodo.org/records/14233292/files/dataset.csv?download=1'
EXPECTED='049455aefb5d7a2114b0ec4896a8b5635ba9fdf461eed282cd85208fad122b0e'

def main():
    (ROOT/'data').mkdir(exist_ok=True); (ROOT/'artifacts').mkdir(exist_ok=True)
    raw=urllib.request.urlopen(DATA_URL,timeout=120).read()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:
        raise ValueError('Source CSV changed; review the dataset before continuing.')
    (ROOT/'data/source.csv').write_bytes(raw)
    model_id='distilbert/distilbert-base-uncased'
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+model_id,timeout=30))
    provenance=ROOT/'artifacts/provenance.json'
    revision=json.loads(provenance.read_text())['model_revision'] if provenance.exists() else info['sha']
    dest=ROOT/'.cache/distilbert'; dest.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in ['config.json','tokenizer_config.json','tokenizer.json','vocab.txt','model.safetensors']:
        target=dest/name
        if not target.exists():
            url=f'https://huggingface.co/{model_id}/resolve/{revision}/{name}?download=true&fresh={int(time.time())}'
            with urllib.request.urlopen(url,timeout=120) as response, target.with_suffix(target.suffix+'.part').open('wb') as out:
                while chunk:=response.read(1024*1024): out.write(chunk)
            target.with_suffix(target.suffix+'.part').replace(target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
        if provenance.exists():
            expected_files=json.loads(provenance.read_text())['model_file_sha256']
            if hashes[name]!=expected_files[name]:
                raise ValueError('Cached model file checksum does not match recorded provenance: '+name)
        print('Verified downloaded file:',name,flush=True)
    provenance.write_text(json.dumps({'dataset_url':DATA_URL,'doi':'10.5281/zenodo.14233292',
        'version':'v1.0','license':'CC BY 4.0 as stated by creator on Zenodo',
        'attribution':'Muhammad Osama Nusrat (2024), Multi-Class Depression Detection Dataset',
        'csv_sha256':EXPECTED, 'model_id':model_id,'model_revision':revision,'model_license':'Apache-2.0',
        'model_file_sha256':hashes,'verified_on':'2026-11-02'},indent=2),encoding='utf-8')

if __name__=='__main__': main()
