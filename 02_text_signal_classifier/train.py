import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score,ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from signals import ROOT,LABELS,clean_text,duplicate_key,mask_label_terms,LABEL_PATTERN,Embedder

OUT=ROOT/'artifacts'; SEED=42; LIMIT=3000
def dump(name,data): (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def evaluate(y,pred):
    return {'macro_f1':float(f1_score(y,pred,average='macro')),
            'classification_report':classification_report(y,pred,labels=LABELS,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,pred,labels=LABELS).tolist(),'n':len(y)}

def main():
    OUT.mkdir(exist_ok=True)
    provenance=json.loads((OUT/'provenance.json').read_text())
    assert hashlib.sha256((ROOT/'data/source.csv').read_bytes()).hexdigest()==provenance['csv_sha256']
    raw=pd.read_csv(ROOT/'data/source.csv')
    assert list(raw.columns)==['Tweets','Labels']
    assert set(raw.Labels.dropna())==set(LABELS)
    audit={'raw_rows':len(raw),'columns':list(raw.columns),'raw_class_counts':raw.Labels.value_counts().to_dict(),
           'missing':raw.isna().sum().to_dict(),'user_ids':False,'timestamps':False,'behavioral_fields':False}
    df=raw.dropna().copy()
    df['text']=df.Tweets.map(clean_text)
    df['key']=df.text.map(duplicate_key)
    df=df.loc[df.key.str.len()>=10]
    conflicts=df.groupby('key').Labels.nunique()
    conflict_keys=conflicts[conflicts>1].index
    audit['conflicting_duplicate_groups']=len(conflict_keys)
    audit['conflicting_rows_removed']=int(df.key.isin(conflict_keys).sum())
    df=df.loc[~df.key.isin(conflict_keys)]
    audit['same_label_duplicates_removed']=int(df.duplicated('key').sum())
    df=df.drop_duplicates('key')
    audit['unique_eligible_rows']=len(df)
    # Fixed stratified CPU subset; not cherry-picked by observed model performance.
    if len(df)>LIMIT:
        ids,_=train_test_split(df.index,train_size=LIMIT,stratify=df.Labels,random_state=SEED)
        df=df.loc[ids].copy()
    train_ids,rest=train_test_split(df.index,test_size=.4,stratify=df.Labels,random_state=SEED)
    val_ids,test_ids=train_test_split(rest,test_size=.5,stratify=df.loc[rest,'Labels'],random_state=SEED)
    splits={'train':train_ids.tolist(),'validation':val_ids.tolist(),'test':test_ids.tolist()}
    audit['split_counts']={k:df.loc[ids,'Labels'].value_counts().to_dict() for k,ids in splits.items()}
    audit['explicit_label_term_counts']={k:int(df.loc[ids,'text'].map(lambda t:bool(LABEL_PATTERN.search(t))).sum()) for k,ids in splits.items()}
    audit['limitations']='Creator uses lexicon labeling; topics/keywords may define labels. No clinical diagnosis or user-disjoint guarantee. Exact normalization duplicates removed; semantic near duplicates can remain. No source text is included in published evaluation artifacts.'
    dump('split_manifest.json',{'seed':SEED,'subset_limit':LIMIT,'source_sha256':provenance['csv_sha256'],'row_indices':splits})
    text={k:df.loc[ids,'text'].tolist() for k,ids in splits.items()}
    y={k:df.loc[ids,'Labels'].to_numpy() for k,ids in splits.items()}
    t0=time.perf_counter()
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=15000,sublinear_tf=True)
    tf={}; tf['train']=vectorizer.fit_transform(text['train'])
    for key in ['validation','test']: tf[key]=vectorizer.transform(text[key])
    similarity=cosine_similarity(tf['test'],tf['train'])
    audit['test_rows_with_train_cosine_at_least_0_90']=int((similarity.max(axis=1)>=.9).sum())
    audit['maximum_train_test_cosine']=float(similarity.max())
    dump('data_audit.json',audit)
    embedding_start=time.perf_counter()
    embedder=Embedder()
    embedding_cache=OUT/'embeddings.npz'
    fingerprint=hashlib.sha256(json.dumps({'splits':splits,'revision':provenance['model_revision'],
       'max_length':128,'pooling':'attention-masked mean including special tokens','normalization':'NFKC lower redact URL/email/mention v1'},sort_keys=True).encode()).hexdigest()
    if embedding_cache.exists() and (OUT/'embedding_metadata.json').exists() and json.loads((OUT/'embedding_metadata.json').read_text())['fingerprint']==fingerprint:
        emb=dict(np.load(embedding_cache)); emb_meta=json.loads((OUT/'embedding_metadata.json').read_text())
    else:
        emb={}; counts={}
        for key in splits:
            emb[key],counts[key]=embedder.encode(text[key],progress=True)
        np.savez_compressed(embedding_cache,**emb)
        emb_meta={'fingerprint':fingerprint,'truncated_counts':counts,'dimension':768,'max_length':128,
                  'pooling':'attention-masked mean, including special tokens','seconds':time.perf_counter()-embedding_start,
                  'frozen':True,'device':'cpu','threads':4,'batch_size':8}
        dump('embedding_metadata.json',emb_meta)
    fitted={}; tuning={}; timings={}
    for name,features in [('tfidf',tf),('distilbert',emb)]:
        candidates=[]
        start=time.perf_counter()
        for c in [.1,1.0,10.0]:
            model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=SEED)
            if name=='distilbert': model=Pipeline([('scale',StandardScaler()),('classifier',model)])
            model.fit(features['train'],y['train'])
            result=evaluate(y['validation'],model.predict(features['validation']))
            candidates.append((result['macro_f1'],c,model,result))
        score,c,model,result=max(candidates,key=lambda item:(item[0],-item[1]))
        tuning[name]={'C':c,'validation':result,'candidate_macro_f1':{str(x[1]):x[0] for x in candidates}}
        fitted[name]=model; timings[name]=time.perf_counter()-start
    dummy=DummyClassifier(strategy='most_frequent').fit(tf['train'],y['train'])
    fitted['dummy']=dummy
    tuning['dummy']={'validation':evaluate(y['validation'],dummy.predict(tf['validation']))}
    winner=max(fitted,key=lambda n:tuning[n]['validation']['macro_f1'])
    dump('selection.json',{'selected_model':winner,'criterion':'validation macro-F1; argmax class decisions','tuning':tuning})
    results={}; all_scores={}
    for name,model in fitted.items():
        features=emb if name=='distilbert' else tf
        pred=model.predict(features['test']); scores=model.predict_proba(features['test'])
        results[name]={'validation':tuning[name]['validation'],'test':evaluate(y['test'],pred),
                       'classifier_fit_seconds':timings.get(name,0)}
        table=pd.DataFrame({'source_row':test_ids,'label':y['test'],'prediction':pred})
        for i,label in enumerate(model.classes_):table['score_'+label]=scores[:,i]
        table.to_csv(OUT/f'{name}_test_predictions.csv',index=False)
        joblib.dump(model,OUT/f'{name}.joblib')
    joblib.dump(vectorizer,OUT/'vectorizer.joblib')
    # Diagnostic on the already-frozen models, not a new selection or tuning step.
    masked=[mask_label_terms(t) for t in text['test']]
    masked_emb,masked_trunc=embedder.encode(masked,progress=True)
    diagnostic={
        'tfidf':evaluate(y['test'],fitted['tfidf'].predict(vectorizer.transform(masked))),
        'distilbert':evaluate(y['test'],fitted['distilbert'].predict(masked_emb)),
        'note':'Masks explicit dataset category/depression terms only, not the full creator lexicon. Distribution shift and residual keywords remain; not a causal estimate.'}
    dump('keyword_mask_diagnostic.json',diagnostic)
    dump('metrics.json',results)
    dump('metadata.json',{'selected_model':winner,'labels':LABELS,'seed':SEED,'max_chars':4000,
        'source_sha256':provenance['csv_sha256'],'model_revision':provenance['model_revision'],
        'embedding_max_tokens':128,'behavioral_features':False,'user_disjoint_split_possible':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(name,result) in zip(axes,results.items()):
        ConfusionMatrixDisplay(np.array(result['test']['confusion_matrix']),display_labels=LABELS).plot(ax=ax,colorbar=False,xticks_rotation=60)
        ax.set_title(name)
    fig.tight_layout();fig.savefig(OUT/'confusion_matrices.png',dpi=150);plt.close(fig)
    print(json.dumps({'selected':winner,'test_macro_f1':{n:r['test']['macro_f1'] for n,r in results.items()},'masked_macro_f1':{n:diagnostic[n]['macro_f1'] for n in ['tfidf','distilbert']}},indent=2))

if __name__=='__main__': main()
