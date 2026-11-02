"""Download the official archive, retaining provenance; never scrape patient accounts."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://archive.ics.uci.edu/static/public/225/ilpd%2Bindian%2Bliver%2Bpatient%2Bdataset.zip'

def main():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        raw = response.read(2_000_000)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        if len(names) != 1:
            raise ValueError('Expected exactly one official CSV.')
        csv = archive.read(names[0])
    digest = hashlib.sha256(csv).hexdigest()
    provenance_path = ROOT / 'artifacts/data_provenance.json'
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if old['csv_sha256'] != digest:
            raise ValueError('Upstream data changed. Review before replacing the recorded dataset version.')
    (data / 'ilpd.csv').write_bytes(csv)
    provenance_path.parent.mkdir(exist_ok=True)
    provenance_path.write_text(json.dumps({
        'source': SOURCE, 'source_page': 'https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset',
        'doi': '10.24432/C5D02C', 'license': 'CC BY 4.0',
        'attribution': 'Ramana, B. and Venkateswarlu, N. ILPD, UCI Machine Learning Repository.',
        'archive_sha256': hashlib.sha256(raw).hexdigest(), 'csv_sha256': digest,
        'archive_member': names[0], 'csv_bytes': len(csv),
        'source_verified_on': '2026-11-02',
        'target': 'Selector 1 = liver patient (positive); 2 = non-liver patient (negative). Not cirrhosis.'
    }, indent=2), encoding='utf-8')
    print('Official CSV saved; SHA256', digest)

if __name__ == '__main__':
    main()
