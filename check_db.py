import sqlite3

conn = sqlite3.connect('data/meeting_analysis.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('All tables:', cursor.fetchall())

cursor.execute('PRAGMA table_info(compliance_report)')
print('Compliance report columns:')
for row in cursor.fetchall():
    print(' ', row)

cursor.execute('SELECT COUNT(*) FROM compliance_report')
print('Total compliance reports:', cursor.fetchone()[0])

cursor.execute('SELECT meeting_id, total_score, score_level FROM compliance_report ORDER BY id DESC LIMIT 20')
print('Latest compliance reports:')
for row in cursor.fetchall():
    print(' ', row)

cursor.execute('SELECT id, title, status, total_score FROM meeting ORDER BY id DESC LIMIT 20')
print('Latest meetings:')
for row in cursor.fetchall():
    print(' ', row)

conn.close()
