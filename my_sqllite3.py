import sqlite3
import json

class SQLiteManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        pass # 这里可以根据需要创建表结构

    def close(self):
        self.conn.close()

class LMarenaSQLiteManager(SQLiteManager):
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS LMarena")
        self.conn.commit()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS LMarena (
                id INTEGER PRIMARY KEY,
                id_set INTEGER
            )
        ''')
        self.conn.commit()
        cursor.close()
        
    def insert(self, id: int, id_set: list):
        cursor = self.conn.cursor()
        cursor.execute(f"INSERT INTO LMarena (id, id_set) VALUES (?, ?)", (id, id_set[0]))
        self.conn.commit()
        cursor.close()

    def search_by_id(self, id: int)-> list:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id_set FROM LMarena WHERE id=?", (id,))
        result = cursor.fetchone()
        cursor.close()
        return [result[0]]

class ClassificationSortedSQLiteManager(SQLiteManager):
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS ClassificationSorted")
        self.conn.commit()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS ClassificationSorted (
                id INTEGER PRIMARY KEY,
                response TEXT
            )
        ''')
        self.conn.commit()
        cursor.close()

    def insert(self, id: int, response: list):
        cursor = self.conn.cursor()
        cursor.execute(f"INSERT INTO ClassificationSorted (id, response) VALUES (?, ?)", (id, response[0]))
        self.conn.commit()
        cursor.close()

    def search_by_id(self, id: int)-> list:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT response FROM ClassificationSorted WHERE id=?", (id,))
        result = cursor.fetchone()
        cursor.close()
        return [result[0]]

class SearchQueriesSQLiteManager(SQLiteManager):
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS SearchQueries")
        self.conn.commit()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS SearchQueries (
                id INTEGER PRIMARY KEY,
                id_set INTEGER,
                cluster_id INTEGER
            )
        ''')
        self.conn.commit()
        cursor.close()

    def insert(self, id: int, values: list):
        cursor = self.conn.cursor()
        cursor.execute(f"INSERT INTO SearchQueries (id, id_set, cluster_id) VALUES (?, ?, ?)", (id, values[0], values[1]))
        self.conn.commit()
        cursor.close()

    def search_by_id(self, id: int)-> list[int, int]:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id_set, cluster_id FROM SearchQueries WHERE id=?", (id,))
        result = cursor.fetchone()
        cursor.close()
        return [result[0], result[1]]

class vcache_hit_record_SQLiteManager(SQLiteManager):
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS vcache_hit_record")
        self.conn.commit()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS vcache_hit_record (
                id INTEGER PRIMARY KEY,
                s_vals TEXT NOT NULL,      -- JSON数组格式
                c_vals TEXT NOT NULL       -- JSON数组格式
            )
        ''')
        self.conn.commit()
        cursor.close()

    def add_or_update(self, id: int, s_vals: list, c_vals: list):
        cursor = self.conn.cursor()
        cursor.execute(f'''
            INSERT OR REPLACE INTO vcache_hit_record (id, s_vals, c_vals) 
            VALUES (?, ?, ?)
        ''', (id, json.dumps(s_vals), json.dumps(c_vals)))
        self.conn.commit()
        cursor.close()

    def search_by_id(self, id: int)-> tuple[list, list]:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT s_vals, c_vals FROM vcache_hit_record WHERE id=?", (id,))
        result = cursor.fetchone()
        cursor.close()
        s_vals = json.loads(result[0])
        c_vals = json.loads(result[1])
        return s_vals, c_vals
    
    def get_all_records(self) -> list[tuple[int, list, list]]:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id, s_vals, c_vals FROM vcache_hit_record")
        results = cursor.fetchall()
        cursor.close()
        records = []
        for row in results:
            id = row[0]
            s_vals = json.loads(row[1])
            c_vals = json.loads(row[2])
            records.append((id, s_vals, c_vals))
        return records