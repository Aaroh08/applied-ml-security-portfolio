import json
from pathlib import Path
import pymysql

ROOT=Path(__file__).resolve().parent
def config():return json.loads((ROOT/'runtime/config.json').read_text())
def connection():
    c=config()
    if c['db_host']!='127.0.0.1':raise ValueError('This prototype permits only a loopback database.')
    return pymysql.connect(host=c['db_host'],port=c['db_port'],user=c['db_user'],password=c['db_password'],
       database=c['db_name'],charset='utf8mb4',autocommit=True,connect_timeout=3,
       read_timeout=5,write_timeout=5,cursorclass=pymysql.cursors.DictCursor)

def initialize_schema():
    with connection() as db,db.cursor() as cur:
        cur.execute('''CREATE TABLE IF NOT EXISTS security_events (
          id BIGINT PRIMARY KEY AUTO_INCREMENT, request_id CHAR(32) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, method VARCHAR(8) NOT NULL,
          route VARCHAR(32) NOT NULL, decision VARCHAR(12) NOT NULL, reason VARCHAR(32) NOT NULL,
          model_score DOUBLE NOT NULL, rule_hits TEXT NOT NULL, http_status INT NULL,
          duration_ms DOUBLE NULL, request_summary TEXT NOT NULL)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS products (
          id INT PRIMARY KEY, name VARCHAR(100) NOT NULL, price DECIMAL(8,2) NOT NULL)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS notes (
          id BIGINT PRIMARY KEY AUTO_INCREMENT, body VARCHAR(200) NOT NULL)''')
        cur.executemany('INSERT IGNORE INTO products (id,name,price) VALUES (%s,%s,%s)',
                       [(1,'Blue notebook',3.5),(2,'Green folder',2.0),(3,"O'Reilly reference book",25)])

class EventStore:
    def health(self):
        with connection() as db,db.cursor() as cur:cur.execute('SELECT 1')
    def begin(self,event):
        with connection() as db,db.cursor() as cur:
            cur.execute('''INSERT INTO security_events
                (request_id,method,route,decision,reason,model_score,rule_hits,request_summary)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
                (event['request_id'],event['method'],event['route'],event['decision'],event['reason'],
                 event['model_score'],json.dumps(event['rule_hits']),json.dumps(event['request_summary'])))
            return cur.lastrowid
    def finish(self,event_id,status,duration_ms):
        with connection() as db,db.cursor() as cur:
            cur.execute('UPDATE security_events SET http_status=%s,duration_ms=%s WHERE id=%s',
                        (status,duration_ms,event_id))
    def recent(self):
        with connection() as db,db.cursor() as cur:
            cur.execute('SELECT * FROM security_events ORDER BY id DESC LIMIT 30');rows=cur.fetchall()
            for row in rows:
                row['created_at']=row['created_at'].isoformat()
                row['rule_hits']=json.loads(row['rule_hits']);row['request_summary']=json.loads(row['request_summary'])
            cur.execute("SELECT COUNT(*) total, COALESCE(SUM(decision='block'),0) blocked, COALESCE(SUM(http_status>=500),0) errors FROM security_events")
            summary={k:int(v) for k,v in cur.fetchone().items()}
        return {'summary':summary,'events':rows}
