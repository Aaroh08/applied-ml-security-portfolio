"""Transparent defensive features; intended only for the bundled localhost demo."""
import html
import re
import unicodedata
from urllib.parse import unquote_plus
import numpy as np

FEATURES=['length','single_quotes','double_quotes','angle_brackets','semicolons','equals',
          'percent_signs','digits_ratio','sql_keywords','script_keywords','comment_markers','parentheses']
RULES={
 'sql_union_select':re.compile(r'\bunion\s+(?:all\s+)?select\b'),
 'sql_boolean_tautology':re.compile(r"\b(?:or|and)\s+(?:['\"]?\w+['\"]?)\s*=\s*(?:['\"]?\w+['\"]?)"),
 'sql_stacked_statement':re.compile(r';\s*(?:drop|delete|insert|update)\b'),
 'xss_script_tag':re.compile(r'<\s*/?\s*script\b'),
 'xss_event_handler':re.compile(r'<[^>]*\bon\w+\s*='),
 'xss_javascript_uri':re.compile(r'javascript\s*:')
}

def canonicalize(value):
    value=unicodedata.normalize('NFKC',value)
    for _ in range(2):value=unquote_plus(value)
    return ' '.join(html.unescape(value).lower().split())

def rule_hits(value):
    text=canonicalize(value)
    return [name for name,pattern in RULES.items() if pattern.search(text)]

def features(values):
    rows=[]
    for value in values:
        text=canonicalize(value)
        rows.append([len(text),text.count("'"),text.count('"'),text.count('<')+text.count('>'),
          text.count(';'),text.count('='),value.count('%'),sum(c.isdigit() for c in text)/max(len(text),1),
          len(re.findall(r'\b(?:select|union|from|where|or|and|drop|delete|insert|update|sleep|benchmark)\b',text)),
          len(re.findall(r'\b(?:script|javascript|alert|onerror|onload|document|cookie|svg)\b',text)),
          text.count('--')+text.count('/*')+text.count('*/'),text.count('(')+text.count(')')])
    return np.asarray(rows,dtype=float)

class Detector:
    def __init__(self,model,threshold):self.model,self.threshold=model,threshold
    def inspect(self,value):
        hits=rule_hits(value)
        score=float(self.model.predict_proba(features([value]))[0,1])
        return {'blocked':bool(hits or score>=self.threshold),'rule_hits':hits,'model_score':score,
                'threshold':self.threshold,'reason':'rule' if hits else 'model' if score>=self.threshold else 'allow'}
