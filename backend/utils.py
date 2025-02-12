import yaml
import os
import sqlite3
import psycopg2

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def initialize_database(db_url):
    if "sqlite" in db_url:
        conn = sqlite3.connect(db_url.replace("sqlite:///", ""))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                output_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    else:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_logs (
                id SERIAL PRIMARY KEY,
                prompt TEXT,
                output_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    conn.commit()
    conn.close()
