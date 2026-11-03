"""Generate declared synthetic traffic with template-disjoint evaluation."""
import hashlib
import json
from pathlib import Path
import random
from urllib.parse import quote
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_auc_score
from detector import FEATURES,features,rule_hits

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts'
# Every string is an authored test payload. They are never sent to an external target.
TEMPLATES={
 'train':{
  'benign_catalog':(0,'notebook {n}'), 'benign_name':(0,"O'Reilly volume {n}"),
  'benign_sentence':(0,'please select a blue or green folder {n}'),
  'benign_math':(0,'math {n} < 999 and 999 > {n}'),
  'benign_note':(0,'meeting notes for item {n}; bring a pen'),
  'sql_union':(1,"{n}' UNION SELECT name FROM products --"),
  'sql_boolean':(1,"{n}' OR 1=1 --"),
  'sql_stacked':(1,"{n}'; DROP TABLE imaginary_table; --"),
  'xss_script':(1,"<script>alert({n})</script>"),
  'xss_event':(1,"<img src=x onerror=alert({n})>"),
  'xss_uri':(1,"javascript:alert({n})")},
 'validation':{
  'benign_stationery':(0,'paper pack {n}'), 'benign_punctuation':(0,"customer's order #{n}: ready"),
  'benign_sql_book':(0,'SQL SELECT examples chapter {n}'),
  'benign_code_words':(0,'document and cookie reference page {n}'),
  'sql_union_all':(1,"{n} UNION ALL SELECT price FROM products"),
  'sql_quoted_boolean':(1,"{n}' OR 'x'='x' --"),
  'xss_body':(1,"<body onload=alert({n})>"),
  'xss_capital_script':(1,"<SCRIPT >alert({n})</SCRIPT>")},
 'test':{
  'benign_gift':(0,'gift card {n} for a friend'),
  'benign_parentheses':(0,'folder ({n}) in blue'),
  'benign_html_tutorial':(0,'how to write <script> tags chapter {n}'),
  'benign_equals':(0,'size={n}; color=blue'),
  'benign_percent':(0,'discount {n}% on notebooks'),
  'benign_double_quote':(0,'book "SQL" edition {n}'),
  'sql_whitespace':(1,"{n}'\tUNION\nSELECT id, name FROM products --"),
  'sql_update':(1,"{n}'; UPDATE imaginary_table SET item=1; --"),
  'sql_obfuscated_comment':(1,"{n}' UNI/**/ON SEL/**/ECT name FROM products --"),
  'xss_svg':(1,"<svg onload=alert({n})></svg>"),
  'xss_link':(1,"<a href='javascript:alert({n})'>sample</a>"),
  'xss_entity':(1,"&#60;script&#62;alert({n})&#60;/script&#62;")}}

def generate():
    rng=random.Random(42);rows=[]
    for split,groups in TEMPLATES.items():
        for family,(label,template) in groups.items():
            for i in range(30):
                n=rng.randrange(1000,999999);text=template.format(n=n)
                if i%3==1:text=quote(text,safe='')
                if i%3==2:text=quote(quote(text,safe=''),safe='')
                rows.append({'split':split,'family':family,'label':label,'payload':text})
    return pd.DataFrame(rows)

def report(y,pred,scores=None):
    matrix=confusion_matrix(y,pred,labels=[0,1])
    return {'n':len(y),'classification_report':classification_report(y,pred,output_dict=True,zero_division=0),
            'confusion_matrix':matrix.tolist(),'false_positives':int(matrix[0,1]),'false_negatives':int(matrix[1,0]),
            'macro_f1':float(f1_score(y,pred,average='macro')),
            'roc_auc':float(roc_auc_score(y,scores)) if scores is not None else None}

def main():
    OUT.mkdir(exist_ok=True);df=generate();df.to_csv(OUT/'synthetic_traffic.csv',index=False)
    assert not df.payload.duplicated().any()
    parts={k:df[df.split==k] for k in TEMPLATES}
    model=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=1)
    model.fit(features(parts['train'].payload),parts['train'].label)
    val=model.predict_proba(features(parts['validation'].payload))[:,1]
    threshold=float(max(np.linspace(.1,.9,81),key=lambda t:(f1_score(parts['validation'].label,val>=t,average='macro'),-abs(t-.5))))
    results={}
    for split,frame in parts.items():
        scores=model.predict_proba(features(frame.payload))[:,1]
        rules=np.array([bool(rule_hits(x)) for x in frame.payload]);rf=scores>=threshold;combined=rules|rf
        results[split]={'rf':report(frame.label,rf,scores),'rules':report(frame.label,rules),'combined':report(frame.label,combined)}
        pred=frame.copy();pred['model_score']=scores;pred['rules_block']=rules;pred['rf_block']=rf;pred['combined_block']=combined
        pred.to_csv(OUT/f'{split}_predictions.csv',index=False)
    joblib.dump(model,OUT/'forest.joblib')
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2))
    (OUT/'metadata.json').write_text(json.dumps({'seed':42,'threshold':threshold,'features':FEATURES,
        'source':'Authored synthetic templates in train.py; no real traffic or external probes',
        'split_policy':'Disjoint named template families; variants remain within family split',
        'family_counts':{k:len(v) for k,v in TEMPLATES.items()},'rows':df.groupby('split').size().to_dict(),
        'csv_sha256':hashlib.sha256((OUT/'synthetic_traffic.csv').read_bytes()).hexdigest(),
        'limitations':'Synthetic patterns overlap feature/rule design. Metrics do not estimate production detection. Benign code examples deliberately test false positives.'},indent=2))
    print(json.dumps({'threshold':threshold,'test':results['test']},indent=2))

if __name__=='__main__':main()
