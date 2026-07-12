import sqlite3
conn = sqlite3.connect('data/meeting_analysis.db')
cursor = conn.cursor()
cursor.execute('SELECT id, title, status, substr(summary, 1, 100) FROM meeting ORDER BY id DESC LIMIT 5')
for row in cursor.fetchall():
    print(f'ID: {row[0]}, Title: {row[1]}, Status: {row[2]}, Summary: {row[3][:80]}')
conn.close()
