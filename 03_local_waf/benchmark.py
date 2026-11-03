"""Sequential loopback benchmark; never accepts external targets."""
import json
import time
from pathlib import Path
import numpy as np
import requests

results={}
for name,url in [('backend','http://127.0.0.1:5091/search'),('gateway','http://127.0.0.1:5053/app/search')]:
    with requests.Session() as s:
        for _ in range(5):s.get(url,params={'q':'notebook'},timeout=5).raise_for_status()
        durations=[];start=time.perf_counter()
        for _ in range(50):
            before=time.perf_counter();r=s.get(url,params={'q':'notebook'},timeout=5);r.raise_for_status()
            assert r.json()['count']==1
            durations.append((time.perf_counter()-before)*1000)
        results[name]={'requests':50,'p50_ms':float(np.percentile(durations,50)),'p95_ms':float(np.percentile(durations,95)),
                       'sequential_requests_per_second':50/(time.perf_counter()-start),'durations_ms':durations}
results['scope']='Single Windows CPU machine; sequential localhost GET with persistent client session; gateway includes MySQL event writes. Not a load or production benchmark.'
Path('artifacts/latency.json').write_text(json.dumps(results,indent=2))
print(json.dumps({k:v for k,v in results.items() if k=='scope' or isinstance(v,dict)},indent=2))
